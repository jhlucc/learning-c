#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TensorRT-YOLO RTSP Multi-Stream Detection System
Designed for Driver Monitoring
"""

import cv2
import numpy as np
import threading
import time
import queue
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from tensorrt_yolo.infer import DetectModel, InferOption
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriverStatus:
    """Driver status data structure"""
    stream_id: int
    timestamp: float
    is_yawning: bool = False
    is_drowsy: bool = False
    is_distracted: bool = False
    is_phone_using: bool = False
    face_detected: bool = False
    eye_closed_duration: float = 0.0
    yawn_duration: float = 0.0
    # 0: Normal, 1: Reminder, 2: Warning, 3: Danger
    alert_level: int = 0


class RTSPDetectionSystem:
    """RTSP multi-stream detection system"""

    def __init__(self, model_path: str, num_streams: int = 4):
        """
        Initialize detection system

        Args:
            model_path: path to TensorRT engine file
            num_streams: number of streams
        """
        self.num_streams = num_streams

        # Initialize model
        logger.info(f"Loading model: {model_path}")
        option = InferOption()
        option.enable_swap_rb()
        option.enable_performance_report()

        # Base model
        self.main_model = DetectModel(model_path, option)

        # Clone one context per stream
        self.models: List[DetectModel] = [self.main_model]
        for _ in range(1, num_streams):
            self.models.append(self.main_model.clone())

        # Stream management
        self.streams: List[Dict] = []
        self.stream_threads: List[threading.Thread] = []
        # Queues for detection worker (producer=reader, consumer=detection)
        self.frame_queues: List[queue.Queue] = [queue.Queue(maxsize=5) for _ in range(num_streams)]

        # Latest-frame buffers for display (read-only for display)
        self.latest_frames: List[Optional[Tuple[np.ndarray, float]]] = [None for _ in range(num_streams)]
        self.latest_locks: List[threading.Lock] = [threading.Lock() for _ in range(num_streams)]

        # Status tracking
        self.driver_status: List[DriverStatus] = [DriverStatus(i, time.time()) for i in range(num_streams)]
        self.status_history: List[List[Dict]] = [[] for _ in range(num_streams)]

        # Control flags
        self.running = False
        self.recording = False
        self.display_mode = 'grid'  # grid, single, alert

        # Video writers
        self.video_writers: Dict[int, cv2.VideoWriter] = {}

        # Alert callbacks
        self.alert_callbacks = []

    def add_rtsp_stream(self, rtsp_url: str, stream_id: Optional[int] = None):
        """
        Add an RTSP stream

        Args:
            rtsp_url: RTSP URL
            stream_id: optional stream ID
        """
        if stream_id is None:
            stream_id = len(self.streams)

        logger.info(f"Adding stream {stream_id}: {rtsp_url}")
        self.streams.append({
            'id': stream_id,
            'url': rtsp_url,
            'cap': None,
            'connected': False,
            'retry_count': 0
        })

    def connect_stream(self, stream_info: dict) -> bool:
        """Connect to an RTSP stream"""
        try:
            logger.info(f"Connecting to stream {stream_info['id']}: {stream_info['url']}")
            # Use FFMPEG backend for robustness
            cap = cv2.VideoCapture(stream_info['url'], cv2.CAP_FFMPEG)

            # Latency-related tuning
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce buffering latency
            except Exception:
                pass
            try:
                cap.set(cv2.CAP_PROP_FPS, 25)
            except Exception:
                pass

            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    stream_info['cap'] = cap
                    stream_info['connected'] = True
                    stream_info['retry_count'] = 0
                    logger.info(f"Stream {stream_info['id']} connected successfully")
                    return True

        except Exception as e:
            logger.error(f"Failed to connect stream {stream_info['id']}: {e}")

        stream_info['connected'] = False
        stream_info['retry_count'] += 1
        return False

    def stream_reader(self, stream_info: dict):
        """Reader thread for one stream"""
        stream_id = stream_info['id']
        retry_delay = 5  # seconds

        while self.running:
            # Ensure connection
            if not stream_info['connected']:
                if stream_info['retry_count'] < 10:
                    logger.warning(f"Stream {stream_id} reconnecting... (attempt {stream_info['retry_count']})")
                    if self.connect_stream(stream_info):
                        stream_info['retry_count'] = 0
                    else:
                        time.sleep(retry_delay)
                        continue
                else:
                    logger.error(f"Stream {stream_id} max retries reached")
                    break

            # Read frames
            try:
                cap = stream_info['cap']
                if cap:
                    ret, frame = cap.read()

                    if ret:
                        ts = time.time()

                        # Draw stream header info *on the frame*
                        self._add_stream_info(frame, stream_id, ts)

                        # Update latest-frame buffer for display (no queue conflict)
                        with self.latest_locks[stream_id]:
                            self.latest_frames[stream_id] = (frame.copy(), ts)

                        # Enqueue for detection; drop oldest when full
                        try:
                            if self.frame_queues[stream_id].full():
                                self.frame_queues[stream_id].get_nowait()
                            self.frame_queues[stream_id].put((frame, ts), block=False)
                        except Exception:
                            pass
                    else:
                        # Mark disconnected and close
                        stream_info['connected'] = False
                        cap.release()

            except Exception as e:
                logger.error(f"Stream {stream_id} read error: {e}")
                stream_info['connected'] = False

    def _add_stream_info(self, frame: np.ndarray, stream_id: int, timestamp: float):
        """Overlay stream info on a frame"""
        h, w = frame.shape[:2]

        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)

        # Channel ID
        cv2.putText(frame, f"CH-{stream_id + 1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Timestamp
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        cv2.putText(frame, time_str, (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Status text
        status = self.driver_status[stream_id]
        status_text = self._get_status_text(status)
        color = self._get_status_color(status.alert_level)
        cv2.putText(frame, status_text, (w - 300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _get_status_text(self, status: DriverStatus) -> str:
        """Human-readable status text"""
        if status.is_phone_using:
            return "Phone use"
        elif status.is_yawning:
            return "Yawning"
        elif status.is_drowsy:
            return "Drowsy"
        elif status.is_distracted:
            return "Distracted"
        elif status.face_detected:
            return "Normal"
        else:
            return "No face detected"

    def _get_status_color(self, alert_level: int) -> tuple:
        """Color by alert level"""
        colors = [
            (0, 255, 0),    # Green - Normal
            (0, 255, 255),  # Yellow - Reminder
            (0, 165, 255),  # Orange - Warning
            (0, 0, 255)     # Red - Danger
        ]
        return colors[min(alert_level, 3)]

    def detect_driver_status(self, frame: np.ndarray, stream_id: int) -> DriverStatus:
        """
        Run detection for one frame

        Args:
            frame: input BGR frame
            stream_id: stream index

        Returns:
            DriverStatus
        """
        # One cloned context per stream
        model = self.models[stream_id]
        result = model.predict(frame)  # returns DetectRes or iterable, depending on your package

        # Update/reset status
        status = self.driver_status[stream_id]
        status.timestamp = time.time()
        status.face_detected = False
        status.is_yawning = False
        status.is_drowsy = False
        status.is_distracted = False
        status.is_phone_using = False

        # Parse results (defensive against different result shapes)
        try:
            if isinstance(result, list):
                detections = result
            elif hasattr(result, '__iter__') and result.__class__.__name__ != 'DetectRes':
                detections = list(result)
            else:
                detections = []
                if hasattr(result, 'num') and getattr(result, 'num') > 0:
                    if hasattr(result, 'boxes'):
                        for i in range(result.num):
                            detections.append(result.boxes[i])
                    elif hasattr(result, 'left'):
                        detections = [result]
                elif hasattr(result, 'detections'):
                    detections = result.detections
                elif hasattr(result, 'results'):
                    detections = result.results

            for det in detections:
                # class id
                if hasattr(det, 'class_id'):
                    class_id = det.class_id
                elif hasattr(det, 'cls'):
                    class_id = det.cls
                else:
                    class_id = 0

                # score
                if hasattr(det, 'confidence'):
                    confidence = det.confidence
                elif hasattr(det, 'conf'):
                    confidence = det.conf
                elif hasattr(det, 'score'):
                    confidence = det.score
                else:
                    confidence = 0.0

                # map class id to driver status (adjust per your labels)
                if class_id == 0:      # face
                    status.face_detected = True
                elif class_id == 1:    # yawn
                    status.is_yawning = True
                    status.yawn_duration += 0.04  # assume 25 fps
                elif class_id == 2:    # eyes closed
                    status.is_drowsy = True
                    status.eye_closed_duration += 0.04
                elif class_id == 3:    # phone
                    status.is_phone_using = True
                elif class_id == 4:    # distracted
                    status.is_distracted = True

                # bbox
                if hasattr(det, 'left'):
                    x1, y1 = int(det.left), int(det.top)
                    x2, y2 = int(det.right), int(det.bottom)
                elif hasattr(det, 'x1'):
                    x1, y1 = int(det.x1), int(det.y1)
                    x2, y2 = int(det.x2), int(det.y2)
                elif hasattr(det, 'bbox') and len(det.bbox) == 4:
                    x1, y1, x2, y2 = map(int, det.bbox)
                else:
                    continue

                color = self._get_status_color(status.alert_level)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"Class {class_id}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        except Exception as e:
            logger.debug(f"Detection parsing error: {e}")
            # Ignore and continue

        # Update alert level, record history
        self._update_alert_level(status)

        self.status_history[stream_id].append({
            'timestamp': status.timestamp,
            'alert_level': status.alert_level,
            'is_yawning': status.is_yawning,
            'is_drowsy': status.is_drowsy,
            'is_distracted': status.is_distracted,
            'is_phone_using': status.is_phone_using
        })
        # Trim history
        if len(self.status_history[stream_id]) > 1000:
            self.status_history[stream_id] = self.status_history[stream_id][-500:]

        return status

    def _update_alert_level(self, status: DriverStatus):
        """Update alert level by rules"""
        if not any([status.is_yawning, status.is_drowsy,
                    status.is_distracted, status.is_phone_using]):
            status.alert_level = 0
            status.eye_closed_duration = 0
            status.yawn_duration = 0
            return

        if status.is_phone_using:
            status.alert_level = 3  # Danger
        elif status.eye_closed_duration > 2.0:
            status.alert_level = 3  # Danger
        elif status.yawn_duration > 3.0:
            status.alert_level = 2  # Warning
        elif status.is_drowsy:
            status.alert_level = 2  # Warning
        elif status.is_distracted:
            status.alert_level = 1  # Reminder

        if status.alert_level >= 2:
            self._trigger_alert(status)

    def _trigger_alert(self, status: DriverStatus):
        """Invoke alert callbacks"""
        for callback in self.alert_callbacks:
            callback(status)

        logger.warning(
            f"ALERT - Stream {status.stream_id}: "
            f"Level {status.alert_level} - {self._get_status_text(status)}"
        )

    def process_streams(self):
        """Start detection worker and display loop"""
        detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        detection_thread.start()

        # Display loop (blocking)
        self._display_worker()

    def _detection_worker(self):
        """Detection worker loop"""
        while self.running:
            for stream_id in range(len(self.streams)):
                try:
                    frame_data = self.frame_queues[stream_id].get_nowait()
                    if frame_data:
                        frame, timestamp = frame_data
                        # Run detection in-place (draw boxes on the same frame)
                        _ = self.detect_driver_status(frame, stream_id)

                        if self.recording:
                            self._record_frame(stream_id, frame)

                except queue.Empty:
                    pass
                except Exception as e:
                    logger.error(f"Detection error on stream {stream_id}: {e}")

            time.sleep(0.01)  # small sleep to reduce CPU spin

    def _display_worker(self):
        """Display worker (UI & mode switching)"""
        while self.running:
            if self.display_mode == 'grid':
                self._display_grid()
            elif self.display_mode == 'single':
                self._display_single()
            elif self.display_mode == 'alert':
                self._display_alerts()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop()
                break
            elif key == ord('r'):
                self.recording = not self.recording
                logger.info(f"Recording: {self.recording}")
            elif key == ord('g'):
                self.display_mode = 'grid'
            elif key == ord('s'):
                self.display_mode = 'single'
            elif key == ord('a'):
                self.display_mode = 'alert'

    def _display_grid(self):
        """
        Grid display without consuming detection queues.
        Read the last frame from latest_frames with locks to avoid flicker.
        """
        frames: List[np.ndarray] = []

        for stream_id in range(len(self.streams)):
            frame_to_show = None
            try:
                with self.latest_locks[stream_id]:
                    if self.latest_frames[stream_id] is not None:
                        frame_to_show, _ = self.latest_frames[stream_id]
            except Exception:
                frame_to_show = None

            if frame_to_show is None:
                frame_to_show = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame_to_show = cv2.resize(frame_to_show, (640, 480))

            frames.append(frame_to_show)

        # Expecting 4 streams: 2x2 grid
        if len(frames) >= 4:
            row1 = np.hstack(frames[:2])
            row2 = np.hstack(frames[2:4])
            grid = np.vstack([row1, row2])
            cv2.imshow("Driver Monitoring System", grid)
        else:
            # Fallback: show whatever we have horizontally
            canvas = np.hstack(frames)
            cv2.imshow("Driver Monitoring System", canvas)

    def _display_single(self):
        """Optional single-view mode (show stream 0)"""
        frame_to_show = None
        try:
            with self.latest_locks[0]:
                if self.latest_frames[0] is not None:
                    frame_to_show, _ = self.latest_frames[0]
        except Exception:
            frame_to_show = None

        if frame_to_show is None:
            frame_to_show = np.zeros((720, 1280, 3), dtype=np.uint8)

        cv2.imshow("Driver Monitoring System - Single", frame_to_show)

    def _display_alerts(self):
        """Optional alert-centric view (simple placeholder)"""
        # You can implement an alert dashboard here if needed
        self._display_grid()

    def _record_frame(self, stream_id: int, frame: np.ndarray):
        """Record frames per stream"""
        if stream_id not in self.video_writers:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"stream_{stream_id}_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writers[stream_id] = cv2.VideoWriter(
                filename, fourcc, 25.0, (frame.shape[1], frame.shape[0])
            )
            logger.info(f"Started recording: {filename}")

        self.video_writers[stream_id].write(frame)

    def start(self):
        """Start the whole system"""
        self.running = True

        # Start reader threads
        for stream_info in self.streams:
            thread = threading.Thread(target=self.stream_reader, args=(stream_info,), daemon=True)
            thread.start()
            self.stream_threads.append(thread)

        logger.info(f"Started {len(self.streams)} stream readers")

        # Start detection + display
        self.process_streams()

    def stop(self):
        """Gracefully stop the system"""
        logger.info("Stopping system...")
        self.running = False

        # Join readers
        for thread in self.stream_threads:
            try:
                thread.join(timeout=2)
            except Exception:
                pass

        # Release captures
        for stream_info in self.streams:
            if stream_info['cap']:
                try:
                    stream_info['cap'].release()
                except Exception:
                    pass

        # Close recorders
        for writer in self.video_writers.values():
            try:
                writer.release()
            except Exception:
                pass

        cv2.destroyAllWindows()
        logger.info("System stopped")


def main():
    """CLI entry"""
    import argparse

    parser = argparse.ArgumentParser(description="TensorRT-YOLO RTSP Multi-Stream Detection System")
    parser.add_argument("--model", required=True, help="Path to TensorRT engine")
    parser.add_argument("--streams", nargs='+', required=True, help="RTSP stream URLs")
    parser.add_argument("--record", action='store_true', help="Enable recording to MP4 files")
    args = parser.parse_args()

    # Create system
    system = RTSPDetectionSystem(args.model, len(args.streams))

    # Add streams
    for i, url in enumerate(args.streams):
        system.add_rtsp_stream(url, i)

    # Optional alert callback
    def alert_callback(status: DriverStatus):
        print(f"⚠️ ALERT - Stream {status.stream_id}: "
              f"Level {status.alert_level} - {system._get_status_text(status)}")

    system.alert_callbacks.append(alert_callback)

    # Enable recording if requested
    if args.record:
        system.recording = True

    # Run
    try:
        print("Starting Driver Monitoring System...")
        print("Controls:")
        print("  q - Quit")
        print("  r - Toggle recording")
        print("  g - Grid view")
        print("  s - Single view")
        print("  a - Alert view")
        system.start()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        system.stop()


if __name__ == "__main__":
    # Example:
    # python test.py --model yolo_driver.engine --streams \
    #   rtsp://127.0.0.1:8554/ch1 rtsp://127.0.0.1:8554/ch2 \
    #   rtsp://127.0.0.1:8554/ch3 rtsp://127.0.0.1:8554/ch4
    main()
