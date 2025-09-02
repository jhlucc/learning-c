
 
 
 C++ × Android（NDK）多模型本地视觉 App 学习与实战计划（12周）

适用目标
- 用 C++ 实现移动端本地推理与图像处理核心，Android 前端用 Kotlin/Java。
- 多模型集成：人脸检测/识别 + 吸烟检测（或其它自定义目标），实时叠加 UI。
- 性能目标：中端机 720p 实时 ≥15–30 FPS；能耗受控；全离线可运行。

默认技术栈（可替换）
- Android Studio Jellyfish（2024.1+），Gradle 8+，minSdk 24/26，target 34
- NDK r26c，CMake 3.22+，C++17
- 前端：Kotlin（CameraX + Compose 或 XML）
- Native：JNI + C++ 核心库（图像预处理、推理、后处理、调度器）
- 推理引擎（默认）：TensorFlow Lite（C++ API）+ GPU/NNAPI Delegate
  - 备选：ONNX Runtime Mobile（C++），MediaPipe（图形化图，偏整体方案）
- 图像处理：OpenCV 4.9（NDK 版）或自研 NEON SIMD 优化
- 性能/调试：Perfetto、Android Studio Profiler、TFLite benchmark_model、Sanitizers

里程碑（Milestones）
- M0 环境就绪：NDK + JNI + CMake 基础通路跑通
- M1 单模型实时：相机取流 → 预处理 → TFLite 模型 → 叠加框
- M2 多模型流水线：人脸检测/识别 + 吸烟检测；基础调度与跟踪
- M3 性能能耗达标：量化/Delegate/零拷贝/流水线并发
- M4 上架级打包：权限与隐私、崩溃与日志、AB 测试勾子、文档与发布

推荐项目结构
- app/（Android 应用层：CameraX、权限、Overlay、设置页）
- native/
  - include/（公共头）
  - src/（核心 C++：image, preprocess, inference, postprocess, tracker, scheduler）
  - third_party/（tflite, opencv, onnxruntime 可选）
  - CMakeLists.txt（导出为一个 or 多个 .so）
- models/（.tflite/.onnx，校验哈希与版本）
- scripts/（模型转换、benchmark、打包脚本）
- docs/（架构、隐私、性能报告）

模型建议与路径
- 人脸检测：BlazeFace/MediaPipe Face Detection（TFLite）或 UltraFace（TFLite/ONNX）
- 人脸识别：MobileFaceNet、FaceNet、ArcFace 轻量化变体（TFLite/ONNX）
- 吸烟检测：两条路线
  - 目标检测：YOLOv5/8n/YOLO-NAS 小模型 → 训练“cigarette/烟雾/手口区域”数据 → 导出 TFLite/ONNX
  - 规则+多信号融合：人脸+手部检测 + 小目标“香烟”/“打火机” → 时序稳定判定（N 帧内近口区域置信度达阈值）
- 性能优先级：先检测（低分辨率/低频），识别/判定在 ROI 上高分辨率执行

12周执行计划（每周6–10小时）

第1周：环境与NDK基础（M0）
- 目标：Android + NDK + CMake 工程跑通，JNI 往返。
- 任务：
  - 新建“Native C++”模板工程；启用 C++17。
  - 在 CMakeLists 添加一个 native-lib，JNI 暴露 hello()，Kotlin 调用并显示。
  - 集成 OpenCV NDK；在 C++ 做一次 YUV → RGB 转换的单元函数（先用 OpenCV）。
- 交付：能在真机上运行，日志中看到 native 输出。

第2周：相机通路与零拷贝思路
- 目标：CameraX + ImageAnalysis 获取 YUV_420_888 帧并传入 native；理解色彩与对齐。
- 任务：
  - Kotlin 侧拿到 ImageProxy/ByteBuffer，使用 GetDirectBufferAddress 传给 C++。
  - C++ 做 YUV → RGB（或直接在 YUV 上做预处理），resize/normalize。
  - 输出简单 FPS、耗时统计；叠加矩形（Kotlin Canvas 或 OpenGL/Compose）。
- 交付：相机预览 + 原始预处理耗时打印（平均/95分位）。

第3周：集成 TFLite，跑通人脸检测（M1）
- 目标：在 C++ 里创建 TFLite Interpreter，载入人脸检测模型，输出框。
- 任务：
  - FlatBufferModel + OpResolver + Interpreter（C++ API），选择 CPU 先跑通。
  - 输入张量形状、mean/std 归一化；输出解析（框、score）。
  - 定时 warm-up，帧抽样（例如每2–3帧跑一次检测）。
- 交付：实时预览叠加人脸框，FPS ≥ 15（720p 可降 540p 输入）。

第4周：人脸识别（向量化匹配）
- 目标：在人脸 ROI 上提特征，做本地匹配（KNN/余弦相似度）。
- 任务：
  - 集成 Face Embedding 模型（112×112），C++ 侧完成对齐/裁剪/归一化。
  - 建立“登记/比对”流程：Kotlin 页面采样 N 张、求均值向量、阈值匹配。
  - 本地加密存储 embedding（如 AES 密钥派生）；隐私提示与开关。
- 交付：Demo 可登记并识别用户；识别延迟 < 200ms。

第5周：吸烟检测（模型准备与集成）
- 目标：第2个模型接入，先拿可用模型，再规划训练。
- 任务：
  - 方案A：用轻量 YOLO TFLite（包含 cigarette 类）先跑通推断 + NMS。
  - 方案B：短期无现成模型，则先“手口区域 + 小物体”多信号规则占位。
  - 预留 datasets/ 与训练脚本接口（roboflow/自采集→标注→导出）。
- 交付：能在预览中检测到“香烟”或给出“可疑吸烟”提示的可运行版本。

第6周：多模型流水线与调度器（M2）
- 目标：同帧/跨帧的多模型执行策略，确保实时性。
- 任务：
  - 设计调度器：主线程取流；Worker 池分配推理任务；ROI 复用；帧率分级。
  - 策略：人脸检测低频 + 跟踪维持；吸烟检测在口鼻 ROI 较高频；负载自适应。
  - 统一时序戳与结果总线，保证跨模型对齐。
- 交付：双模型并行/交错运行，界面稳定显示，丢帧受控。

第7周：多目标跟踪与事件判定
- 目标：让框稳定，输出“吸烟事件”而非单帧提示。
- 任务：
  - 实现轻量跟踪（SORT：卡尔曼+匈牙利匹配），跟踪 ID 稳定。
  - 事件判定：在 N 连续帧内，人脸 ID 的口部邻域出现高置信度“香烟”→触发。
  - 误报抑制：进入/离开门限、时间窗口、角度/遮挡规则。
- 交付：事件流（开始/持续/结束），UI 与日志可回放。

第8周：性能优化 I（量化与 Delegate）（M3）
- 目标：把时延/帧率提升到目标区间。
- 任务：
  - 使用 TFLite int8 动态/全整量化模型；校准与精度对比。
  - 尝试 GPU Delegate（OpenGL/Vulkan）与 NNAPI Delegate；按设备白名单选择。
  - 预分配 Tensor 与工作缓冲；避免重复拷贝；流水线并发。
- 交付：中端机 720p ≥ 20–30 FPS（按机型记录），功耗下降。

第9周：性能优化 II（能耗与热）
- 目标：长时运行稳定不过热。
- 任务：
  - Perfetto 采集帧时间线与 CPU/GPU 利用率；热节流监测。
  - 动态分辨率/帧率/模型频率自适应；前后台策略。
  - 离线批处理模式（用户触发/插电+WiFi时段）。
- 交付：30 分钟连续运行不过热降频或自动降级策略有效。

第10周：工程质量与异常处理
- 目标：健壮性、容错、隐私合规。
- 任务：
  - 对相机中断/旋转/权限丢失做恢复；处理不同摄像头朝向与畸变。
  - NDK 层错误边界：返回码与异常屏障；日志统一（spdlog/安卓日志）。
  - 隐私弹窗与本地处理声明；模型与数据版本化。
- 交付：QA 清单通过；崩溃率低；异常路径可测。

第11周：测试与持续集成
- 目标：可重复构建与自动化验证。
- 任务：
  - NDK 侧用 GoogleTest（android-ndk-r25+支持）跑算法单测（NMS、几何变换等）。
  - 仪器化测试：CameraX 通路冒烟、权限、生命周期。
  - 基准脚本：不同模型/分辨率/Delegate 的时延-精度曲线。
- 交付：CI（本地或云端）一键构建，产物与基准报告归档。

第12周：发布打磨（M4）
- 目标：打包可分发的 MVP。
- 任务：
  - 分渠道构建（无网络/带网络可选），ABI 分包（arm64-v8a/armeabi-v7a）。
  - 首启引导、设置页（性能档位/模型切换/隐私开关）。
  - README、技术白皮书（架构/性能/隐私）、问题排查手册。
- 交付：可安装 APK/AAB、文档齐全、演示视频。

关键工程要点（踩坑预警）
- 图像格式：YUV_420_888 三个Plane stride不一致；避免逐像素拷贝，尽量直接使用 ByteBuffer + 原位处理。
- 坐标系/旋转：相机传感器方向与预览方向不同，统一在一个地方处理旋转/镜像。
- 零拷贝：能不转 RGB 就别转；很多模型支持 NV12/NV21 可自适配（或在 C++ 做轻量转）。
- 多模型调度：不要所有模型每帧都跑。按“检测低频 + 跟踪高频 + ROI 高分辨率”的原则分层。
- 设备差异：不同厂商 NNAPI/GPU Delegate 行为不同，做能力探测与回退。
- 线程与生命周期：相机回调线程与 JNI 调度线程分离；Native 全局单例注意析构顺序。

可选替代与升级路线
- 引擎改 ONNX Runtime Mobile：便于统一 PC 训练（PyTorch/ONNX）到端侧；体积略大。
- 用 MediaPipe 构建图：快速搭管线 + GPU 加速，但二次开发灵活性略低。
- 自研 NEON 优化：对热点预处理（resize/normalize）手写 SIMD，节约 20–40%。
- Tracker 升级：从 SORT → DeepSORT（需再加轻量 reid），但功耗涨幅较大，谨慎。

数据与训练（吸烟检测）
- 开源数据少，建议自建小型数据集（Roboflow/LabelImg）：
  - 场景/角度/光照多样；负样本（拿笔、喝水）要多。
  - 先训 YOLOv5/8n（320/416），蒸馏或量化感知训练（QAT）提升端侧精度。
  - 导出 TFLite（带 NMS 或后处理放 C++），或导出 ONNX → ORT Mobile。
- 度量：mAP@0.5、漏检率、误报率；端侧门限与时序策略共同调节。

性能目标与验收
- 中端机（骁龙7系/天玑800系）：720p 端到端 ≥ 20 FPS；人脸识别 ROI < 150ms；吸烟事件平均延迟 < 500ms。
- 连续运行 30 分钟机身温度 < 42–45℃；功耗与帧率平衡策略生效。
- 误报/漏报：在自建验证集与实测视频上给出混淆矩阵和事件级指标。

学习与参考资源
- Android NDK 官方文档、NDK Samples（camera2、AImageReader）
- TensorFlow Lite Android & C++ API 指南、Delegate 使用手册
- OpenCV Android SDK 教程（颜色空间、相机、绘制叠加）
- Perfetto/Android Studio Profiler（时延/功耗分析）
- 论文/模型：BlazeFace、MobileFaceNet、YOLO 系列（轻量化）

你需要准备的设备与环境
- 一台主力 Android 真机（Android 10+，最好支持 NNAPI/GPU）
- 一台开发机（Windows/macOS/Linux 均可）
- 采样视频（含目标行为与干扰动作）

下一步（今天就能开始）
- 我可以为你生成一个可运行的“相机取流 + NDK + TFLite 占位推断”的模板工程（含 CMake 与 JNI 桥、性能计时与叠加层）。
- 告诉我：你更倾向 TFLite 还是 ONNX Runtime？UI 用 Compose 还是 XML？目标最低机型与分辨率？我据此把模板与第1–2周任务落到代码级别。
 
 
   
 
