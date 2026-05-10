import cv2  # OpenCV：图像处理、窗口显示、按键捕获
import time  # 高精度计时（perf_counter）
import numpy as np  # 数组操作，构造空帧
import msvcrt  # 无窗口模式下的键盘输入检测（Windows）
from ultralytics import YOLO  # YOLO 推理引擎
import threading  # 多线程：Capture 线程 + FrameBuffer 锁
import dxcam  # DXCam 屏幕捕获（Capture 线程内使用）

from mouse_controller import MouseController  # 鼠标瞄准控制器

# ==================== 模型与推理配置 ====================
MODEL_PATH = "mrzh759s.engine"  # ONNX 模型文件路径
DETECTION_CLASSES = [0, 1, 2]  # 需要检测的类别 ID 列表
CONFIDENCE_THRESHOLD = 0.3  # 推理置信度阈值，低于此值的结果被过滤
WARMUP_ITERATIONS = 10  # 模型预热时的推理次数
WARMUP_CONFIDENCE = 0.25  # 预热时使用的置信度阈值

# ==================== 调试窗口配置 ====================
DEBUG_WINDOW_NAME = "YOLO26 Realtime Detection"  # OpenCV 窗口标题
DEBUG_WINDOW_SIZE = (320, 320)  # 调试窗口的初始尺寸
TOPMOST_INTERVAL = 30  # 每隔多少帧强制置顶一次窗口（防止被其他窗口遮盖）


class FrameBuffer:
    """线程安全的单帧缓冲区：Capture 线程写入，主线程读取最新帧"""

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()

    def set(self, frame):
        """Capture 线程调用：无阻塞覆盖写入最新帧"""
        with self._lock:
            self._frame = frame

    def get(self):
        """主线程调用：取走当前最新帧并清空缓冲区，返回 None 表示尚无帧"""
        with self._lock:
            f = self._frame
            self._frame = None
            return f


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


def _find_closest_target(boxes, mouse_x, mouse_y, capture_region, debug=False):
    if len(boxes) == 0:
        return None

    xyxy = boxes.xyxy
    cls_ids = boxes.cls
    centers = (xyxy[:, :2] + xyxy[:, 2:]) / 2.0

    centers_by_cls = {}
    for cls_id in DETECTION_CLASSES:
        mask = cls_ids == cls_id
        if mask.sum() > 0:
            cls_centers = centers[mask]
            centers_by_cls[cls_id] = cls_centers
            if debug:
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


def _render_frame(result, fps, latency_ms, frame_count=0):
    annotated_frame = result.plot()
    if result.boxes is not None:
        boxes = result.boxes
        person_mask = boxes.cls == 2
        if person_mask.sum() > 0:
            xyxy = boxes.xyxy[person_mask]
            centers = ((xyxy[:, :2] + xyxy[:, 2:]) / 2).cpu().numpy().astype(int)
            for cx, cy in centers:
                cv2.circle(annotated_frame, (cx, cy), 4, (0, 0, 255), -1)
    if frame_count % 60 == 0:
        _render_frame._fps = int(fps)
        _render_frame._latency = latency_ms
    cv2.putText(annotated_frame, f"FPS: {_render_frame._fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Time: {_render_frame._latency:.1f} ms", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return annotated_frame

_render_frame._fps = 0
_render_frame._latency = 0.0


def _check_exit(debug):
    """检测用户是否按下退出键

    调试模式下使用 OpenCV 的 waitKey，无窗口模式下使用 msvcrt 检测键盘输入
    """
    if debug:
        return cv2.waitKey(1) & 0xFF == ord('q')
    # Windows 下 msvcrt.kbhit() 为非阻塞按键检测
    return msvcrt.kbhit() and msvcrt.getch().lower() == b'q'


def capture_worker(buffer, capture_region, stop_event):
    """独立线程：持续抓取屏幕帧并写入 FrameBuffer"""
    camera = None
    try:
        camera = dxcam.create(region=capture_region, output_color="BGR")
        camera.start(target_fps=120, video_mode=True)
        print("Capture 线程已启动")
        while not stop_event.is_set():
            frame = camera.get_latest_frame()
            if frame is not None:
                buffer.set(frame)
            else:
                time.sleep(0.001)
    finally:
        if camera is not None:
            camera.stop()
        print("Capture 线程已退出")


def run_detection(capture_region, debug=True, stop_event=None):
    """主检测入口：加载模型 → 预热 → 启动截图线程 → 循环推理 + 显示"""
    model = load_model()
    warmup_model(model, capture_region)

    if debug:
        setup_debug_window()

    buffer = FrameBuffer()
    if stop_event is None:
        stop_event = threading.Event()
    capture_thread = threading.Thread(
        target=capture_worker,
        args=(buffer, capture_region, stop_event),
        daemon=True,
    )
    capture_thread.start()

    print(f"准备开始检测，截取区域: {capture_region}")
    print("按 'q' 键退出...")

    mouse_ctrl = MouseController()
    prev_time = time.perf_counter()
    frame_count = 0

    try:
        while not stop_event.is_set():
            frame_start_time = time.perf_counter()

            frame = buffer.get()
            if frame is None:
                if not capture_thread.is_alive():
                    print("错误：Capture 线程已终止")
                    break
                time.sleep(0.001)
                continue

            frame_count += 1
            fps = 1 / (frame_start_time - prev_time) if (frame_start_time - prev_time) > 0 else 0
            prev_time = frame_start_time

            results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, classes=DETECTION_CLASSES, verbose=False)

            mouse_x, mouse_y = mouse_ctrl.mouse_ctl.position
            boxes = results[0].boxes
            closest_center = _find_closest_target(boxes, mouse_x, mouse_y, capture_region, debug)

            if closest_center:
                mouse_ctrl.update_target(closest_center[0], closest_center[1], capture_region[0], capture_region[1])
            else:
                mouse_ctrl.update_target(None, None, capture_region[0], capture_region[1])

            latency_ms = (time.perf_counter() - frame_start_time) * 1000

            if debug:
                annotated_frame = _render_frame(results[0], fps, latency_ms, frame_count)
                cv2.imshow(DEBUG_WINDOW_NAME, annotated_frame)
                if frame_count % TOPMOST_INTERVAL == 0:
                    cv2.setWindowProperty(DEBUG_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

            if _check_exit(debug):
                break
    finally:
        stop_event.set()
        capture_thread.join(timeout=2)
        if debug:
            cv2.destroyAllWindows()
        mouse_ctrl.stop()


if __name__ == "__main__":
    run_detection((640, 220, 1280, 640))
