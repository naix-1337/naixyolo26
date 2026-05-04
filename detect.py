import cv2  # OpenCV：图像处理、窗口显示、按键捕获
import time  # 高精度计时（perf_counter）
import numpy as np  # 数组操作，构造空帧
import msvcrt  # 无窗口模式下的键盘输入检测（Windows）
from ultralytics import YOLO  # YOLO 推理引擎
from ScreenCapture import capture_screen_iter  # DXCam 屏幕捕获迭代器
from mouse_controller import MouseController  # 鼠标瞄准控制器

# ==================== 模型与推理配置 ====================
MODEL_PATH = "mrzh759s.onnx"  # ONNX 模型文件路径
DETECTION_CLASSES = [0, 1, 2]  # 需要检测的类别 ID 列表
CONFIDENCE_THRESHOLD = 0.3  # 推理置信度阈值，低于此值的结果被过滤
WARMUP_ITERATIONS = 10  # 模型预热时的推理次数
WARMUP_CONFIDENCE = 0.25  # 预热时使用的置信度阈值

# ==================== 调试窗口配置 ====================
DEBUG_WINDOW_NAME = "YOLO26 Realtime Detection"  # OpenCV 窗口标题
DEBUG_WINDOW_SIZE = (320, 320)  # 调试窗口的初始尺寸
TOPMOST_INTERVAL = 30  # 每隔多少帧强制置顶一次窗口（防止被其他窗口遮盖）


def load_model():
    """加载 YOLO 模型并返回模型对象"""
    print("加载 YOLO 模型...")
    return YOLO(MODEL_PATH)


def warmup_model(model, capture_region):
    """用全黑空帧对模型进行预热推理，确保 GPU/CPU 完成初始化

    预热能消除首次推理时的额外延迟（JIT 编译、显存分配等）
    """
    w = capture_region[2] - capture_region[0]
    h = capture_region[3] - capture_region[1]
    dummy_frame = np.zeros((h, w, 3), dtype=np.uint8)
    print("模型预热中，执行 10 次空数据推理...")
    for _ in range(WARMUP_ITERATIONS):
        model.predict(source=dummy_frame, conf=WARMUP_CONFIDENCE, classes=DETECTION_CLASSES, verbose=False)


def setup_debug_window():
    """创建并配置 OpenCV 调试窗口（可调整大小、置顶）"""
    cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DEBUG_WINDOW_NAME, *DEBUG_WINDOW_SIZE)
    cv2.setWindowProperty(DEBUG_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)


def _find_closest_target(boxes, mouse_x, mouse_y, capture_region):
    """在检测结果中按类别分组，找到离鼠标最近的类型 2 目标的中心坐标

    参数:
        boxes: YOLO 检测结果的 boxes 对象
        mouse_x, mouse_y: 当前鼠标屏幕绝对坐标
        capture_region: 截屏区域 (left, top, right, bottom)

    返回:
        (center_x, center_y) 或 None（无目标时）
    """
    if len(boxes) == 0:
        return None

    # 提取边界框坐标和类别 ID
    xyxy = boxes.xyxy
    cls_ids = boxes.cls
    # 向量化计算每个边界框的中心点
    centers = (xyxy[:, :2] + xyxy[:, 2:]) / 2.0

    # 按类别分组，便于后续按类别筛选目标
    centers_by_cls = {}
    for cls_id in DETECTION_CLASSES:
        mask = cls_ids == cls_id
        if mask.sum() > 0:
            cls_centers = centers[mask]
            centers_by_cls[cls_id] = cls_centers
            print(f"📦 类别 {cls_id} 目标数: {len(cls_centers)}, 中心: {[(f'{c[0]:.1f}', f'{c[1]:.1f}') for c in cls_centers]}")
        else:
            centers_by_cls[cls_id] = None

    # 只对类型 2 的目标计算与鼠标的最近距离
    if centers_by_cls[2] is not None:
        cls2_centers = centers_by_cls[2]
        # 将局部坐标转换为屏幕绝对坐标
        abs_centers_x = cls2_centers[:, 0] + capture_region[0]
        abs_centers_y = cls2_centers[:, 1] + capture_region[1]

        # 向量化计算欧几里得距离平方（省去开平方，不影响比较结果）
        dx = abs_centers_x - mouse_x
        dy = abs_centers_y - mouse_y
        dists = dx.pow(2) + dy.pow(2)

        best_idx = dists.argmin().item()
        return cls2_centers[best_idx, 0].item(), cls2_centers[best_idx, 1].item()

    return None


def _render_frame(frame, boxes, fps, latency_ms):
    """在帧上绘制检测框和性能信息，返回标注后的图像

    参数:
        frame: 原始 BGR 帧
        boxes: YOLO 检测结果的 boxes 对象（用于绘制边框）
        fps: 当前帧率
        latency_ms: 推理延迟（毫秒）
    """
    annotated_frame = frame.copy()
    if boxes is not None and len(boxes) > 0:
        # 将边框坐标转换为整数并逐个绘制矩形
        xyxy_np = boxes.xyxy.cpu().numpy().astype(int)
        for (x1, y1, x2, y2) in xyxy_np:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 左上角叠加 FPS 和延迟信息
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Time: {latency_ms:.1f} ms", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return annotated_frame


def _check_exit(debug):
    """检测用户是否按下退出键

    调试模式下使用 OpenCV 的 waitKey，无窗口模式下使用 msvcrt 检测键盘输入
    """
    if debug:
        return cv2.waitKey(1) & 0xFF == ord('q')
    # Windows 下 msvcrt.kbhit() 为非阻塞按键检测
    return msvcrt.kbhit() and msvcrt.getch().lower() == b'q'


def run_detection(capture_region, debug=True):
    """主检测入口：加载模型 → 预热 → 循环截屏推理 → 鼠标控制

    参数:
        capture_region: 截屏区域 (left, top, right, bottom)
        debug: 是否显示 OpenCV 调试窗口
    """
    model = load_model()
    warmup_model(model, capture_region)

    if debug:
        setup_debug_window()

    print(f"准备开始检测，截取区域: {capture_region}")
    print("开始截图和推理，按 'q' 键退出...")

    mouse_ctrl = MouseController()
    prev_time = time.perf_counter()
    frame_count = 0

    for frame in capture_screen_iter(region=capture_region):
        frame_start_time = time.perf_counter()
        frame_count += 1

        # 计算当前帧率（基于两次帧到达的时间间隔）
        fps = 1 / (frame_start_time - prev_time) if frame_start_time - prev_time > 0 else 0
        prev_time = frame_start_time

        # YOLO 推理
        results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, classes=DETECTION_CLASSES, verbose=False)

        # 找出离鼠标最近的目标并更新瞄准器
        mouse_x, mouse_y = mouse_ctrl.mouse_ctl.position
        boxes = results[0].boxes
        closest_center = _find_closest_target(boxes, mouse_x, mouse_y, capture_region)

        if closest_center:
            mouse_ctrl.update_target(closest_center[0], closest_center[1], capture_region[0], capture_region[1])
            print(f"🎯 最近类型2目标的中心坐标: ({closest_center[0]:.1f}, {closest_center[1]:.1f})")
        else:
            mouse_ctrl.update_target(None, None, capture_region[0], capture_region[1])

        # 计算推理延迟
        inference_end_time = time.perf_counter()
        latency_ms = (inference_end_time - frame_start_time) * 1000

        if debug:
            # 绘制检测结果并显示
            annotated_frame = _render_frame(frame, boxes, fps, latency_ms)
            cv2.imshow(DEBUG_WINDOW_NAME, annotated_frame)
            if frame_count % TOPMOST_INTERVAL == 0:
                cv2.setWindowProperty(DEBUG_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

        if _check_exit(debug):
            break

    if debug:
        cv2.destroyAllWindows()
    mouse_ctrl.stop()


if __name__ == "__main__":
    run_detection((640, 220, 1280, 640))
