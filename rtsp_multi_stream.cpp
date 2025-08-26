#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <optional>

#include "trtyolo.hpp"  // TensorRT-YOLO C++ API


// ---- helpers to read fields across versions ----
template <typename T, typename = void>
struct has_member_class_label : std::false_type {};
template <typename T>
struct has_member_class_label<T, std::void_t<decltype(std::declval<const T&>().class_label)>> : std::true_type {};

template <typename T, typename = void>
struct has_member_cls : std::false_type {};
template <typename T>
struct has_member_cls<T, std::void_t<decltype(std::declval<const T&>().cls)>> : std::true_type {};

template <typename T, typename = void>
struct has_member_label : std::false_type {};
template <typename T>
struct has_member_label<T, std::void_t<decltype(std::declval<const T&>().label)>> : std::true_type {};

template <typename B>
int cls_of(const B& b) {
    if constexpr (has_member_class_label<B>::value) return static_cast<int>(b.class_label);
    else if constexpr (has_member_cls<B>::value)    return static_cast<int>(b.cls);
    else if constexpr (has_member_label<B>::value)  return static_cast<int>(b.label);
    else return 0;
}

// ---------- small helpers ----------
struct FrameItem {
    cv::Mat frame;       // BGR
    double  ts{};        // seconds
};

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t cap) : cap_(cap) {}

    // push with drop-oldest when full (non-blocking)
    void push_drop_oldest(T&& v) {
        std::lock_guard<std::mutex> lk(m_);
        if (q_.size() >= cap_) q_.pop();
        q_.push(std::move(v));
    }

    // non-blocking pop; return false if empty
    bool try_pop(T& out) {
        std::lock_guard<std::mutex> lk(m_);
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop();
        return true;
    }

private:
    std::mutex m_;
    std::queue<T> q_;
    size_t cap_;
};

struct LatestFrame {
    cv::Mat frame;  // last displayed frame
    double  ts{};
    bool    has{false};
};

struct DriverStatus {
    // simplified demo fields; extend as you wish
    int    alert_level{0};  // 0..3
    bool   face{false}, yawning{false}, drowsy{false}, distracted{false}, phone{false};
};

// ---------- per-stream context ----------
struct StreamCtx {
    int id{};
    std::string url;
    cv::VideoCapture cap;
    std::atomic<bool> connected{false};
    int retry_count{0};

    // queues & buffers
    BoundedQueue<FrameItem> detq{5};  // detection queue
    LatestFrame latest;                // display-only latest buffer
    std::mutex latest_mtx;

    // model clone for this stream
    std::unique_ptr<trtyolo::DetectModel> model;
    DriverStatus status;
};

// ---------- overlay helpers ----------
static cv::Scalar statusColor(int level) {
    switch(level){
        case 1: return {0,255,255}; // yellow
        case 2: return {0,165,255}; // orange
        case 3: return {0,0,255};   // red
        default:return {0,255,0};   // green
    }
}

static std::string statusText(const DriverStatus& s){
    if (s.phone)      return "Phone use";
    if (s.yawning)    return "Yawning";
    if (s.drowsy)     return "Drowsy";
    if (s.distracted) return "Distracted";
    if (s.face)       return "Normal";
    return "No face";
}

static void drawHeader(cv::Mat& frame, const StreamCtx& S, double ts){
    int w = frame.cols;
    cv::rectangle(frame, {0,0}, {w,50}, {0,0,0}, cv::FILLED);
    cv::putText(frame, "CH-" + std::to_string(S.id+1), {10,30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,0}, 2);
    // timestamp
    std::time_t t = (std::time_t)ts;
    char buf[64]; std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    cv::putText(frame, buf, {150,30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {255,255,255}, 1);
    // status
    auto color = statusColor(S.status.alert_level);
    cv::putText(frame, statusText(S.status), {w-300,30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
}

// ---------- stream reader thread ----------
static bool connect_stream(StreamCtx& S){
    try{
        // Use FFMPEG backend if OpenCV is built with it
        S.cap.open(S.url, cv::CAP_FFMPEG);
        if(!S.cap.isOpened()) return false;
        // low-latency hints
        S.cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        S.cap.set(cv::CAP_PROP_FPS, 25);
        cv::Mat test; if(!S.cap.read(test) || test.empty()) return false;
        S.connected = true; S.retry_count = 0;
        std::cout << "[stream " << S.id << "] connected: " << S.url << std::endl;
        return true;
    }catch(...){
        return false;
    }
}

static void reader_thread(StreamCtx* S, std::atomic<bool>* running){
    const int max_retry = 10;
    while(running->load()){
        if(!S->connected){
            if(S->retry_count >= max_retry){
                std::cerr << "[stream " << S->id << "] max retries reached\n";
                break;
            }
            std::cerr << "[stream " << S->id << "] reconnecting... attempt " << S->retry_count << "\n";
            if(!connect_stream(*S)){
                ++S->retry_count;
                std::this_thread::sleep_for(std::chrono::seconds(5));
                continue;
            }
        }

        cv::Mat frame;
        if(!S->cap.read(frame) || frame.empty()){
            S->connected = false;
            S->cap.release();
            continue;
        }

        const double ts = (double)std::time(nullptr);

        // overlay header on the frame before any use
        drawHeader(frame, *S, ts);

        // update latest (display does NOT consume detection queue)
        {
            std::lock_guard<std::mutex> g(S->latest_mtx);
            S->latest.frame = frame.clone();  // safe copy for UI
            S->latest.ts = ts; S->latest.has = true;
        }

        // push to detection queue; drop oldest when full
        S->detq.push_drop_oldest(FrameItem{frame, ts});
    }
}

// ---------- simple rule for alert level (demo) ----------
static void update_alert(DriverStatus& st){
    if(st.phone) st.alert_level = 3;
    else if(st.drowsy) st.alert_level = 2;
    else if(st.yawning) st.alert_level = 2;
    else if(st.distracted) st.alert_level = 1;
    else st.alert_level = 0;
}

// ---------- detection thread ----------
static void detection_thread(std::vector<StreamCtx>* streams, std::atomic<bool>* running){
    while(running->load()){
        for(auto& S : *streams){
            FrameItem item;
            if(!S.detq.try_pop(item)) continue;

            // run inference on this stream's clone
            try{
                trtyolo::Image timg{ item.frame.data, item.frame.cols, item.frame.rows };
                auto result = S.model->predict(timg);

                // Reset demo status
                S.status = {};

                // Parse result (fields depend on your API; here we follow your examples)
                // Mark face found by default if any det exists
                bool any_det=false;
                if (result.num > 0) {
                    any_det = true;
                    for(int i=0;i<result.num;i++){
                        const auto& b = result.boxes[i];
                        cv::rectangle(item.frame, { (int)b.left,(int)b.top }, { (int)b.right,(int)b.bottom }, statusColor(S.status.alert_level), 2);
                        // map classes to demo flags (adjust to your real label map)
                        int cls = cls_of(b);
                        if (cls == 0) S.status.face = true;
                        if (cls == 1) S.status.yawning = true;
                        if (cls == 2) S.status.drowsy = true;
                        if (cls == 3) S.status.phone  = true;
                        if (cls == 4) S.status.distracted = true;
                    }
                }
                if(any_det) S.status.face = true;

                update_alert(S.status);

                // update latest (with drawings)
                {
                    std::lock_guard<std::mutex> g(S.latest_mtx);
                    S.latest.frame = item.frame.clone();
                    S.latest.ts = item.ts; S.latest.has = true;
                }
            }catch(const std::exception& e){
                std::cerr << "[detect] stream " << S.id << " error: " << e.what() << "\n";
            }catch(...){
                std::cerr << "[detect] stream " << S.id << " unknown error\n";
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// ---------- display (grid) ----------
static void display_loop(std::vector<StreamCtx>* streams, std::atomic<bool>* running){
    const int W = 640, H = 480;
    while(running->load()){
        std::vector<cv::Mat> tiles;
        tiles.reserve(streams->size());

        for(auto& S : *streams){
            cv::Mat show;
            {
                std::lock_guard<std::mutex> g(S.latest_mtx);
                if (S.latest.has) show = S.latest.frame;
            }
            if(show.empty()) show = cv::Mat(H, W, CV_8UC3, cv::Scalar(0,0,0));
            else cv::resize(show, show, {W,H});
            tiles.push_back(show);
        }

        cv::Mat grid;
        if(tiles.size() >= 4){
            cv::Mat row1, row2;
            cv::hconcat(std::vector<cv::Mat>{tiles[0], tiles[1]}, row1);
            cv::hconcat(std::vector<cv::Mat>{tiles[2], tiles[3]}, row2);
            cv::vconcat(row1, row2, grid);
        }else{
            cv::hconcat(tiles, grid);
        }

        cv::imshow("Driver Monitoring System (C++)", grid);
        int k = cv::waitKey(1) & 0xFF;
        if(k=='q' || k==27){
            running->store(false);
            break;
        }
    }
}

// ---------- main ----------
int main(int argc, char** argv){
    if (argc < 6){
        std::cerr << "Usage: " << argv[0] << " <engine> <rtsp1> <rtsp2> <rtsp3> <rtsp4>\n";
        return 1;
    }

    const std::string engine = argv[1];
    std::vector<std::string> urls = { argv[2], argv[3], argv[4], argv[5] };

    // Build base model and options (enableSwapRB & perf report just like your examples)
    trtyolo::InferOption opt;
    opt.enableSwapRB();
    opt.enablePerformanceReport();

    trtyolo::DetectModel base_model(engine, opt);

    // One cloned model per stream
    std::vector<StreamCtx> streams(urls.size());
    for(size_t i=0;i<urls.size();++i){
        streams[i].id = (int)i;
        streams[i].url = urls[i];
        streams[i].model = base_model.clone();
    }

    std::atomic<bool> running{true};

    // start reader threads
    std::vector<std::thread> readers;
    readers.reserve(streams.size());
    for(auto& S : streams){
        readers.emplace_back(reader_thread, &S, &running);
    }

    // start detection thread
    std::thread detthr(detection_thread, &streams, &running);

    // display loop (blocking)
    display_loop(&streams, &running);

    // join
    running.store(false);
    for(auto& t : readers) if(t.joinable()) t.join();
    if(detthr.joinable()) detthr.join();

    // report perf for each clone (optional)
    for(auto& S : streams){
        auto [thr, gpulat, cpulat] = S.model->performanceReport();
        std::cout << "[stream " << S.id << "] Throughput: " << thr
                  << ", GPU Latency: " << gpulat
                  << ", CPU Latency: " << cpulat << "\n";
    }
    return 0;
}
