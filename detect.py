import msvcrt
import time

import cv2
import numpy as np

from mouse_controller import MouseController
from ScreenCapture import capture_screen_iter
from ultralytics import YOLO


def run_detection(capture_region, debug=True):
    print("加载 YOLO 模型...")
    # 使用现有的模型
    model = YOLO("yolo26s_1650ti_sim.onnx")

    # 预热推理，跑 50 次黑屏废数据
    print("模型预热中，执行 50 次空数据推理...")
    w = capture_region[2] - capture_region[0]
    h = capture_region[3] - capture_region[1]
    dummy_frame = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(50):
        model.predict(source=dummy_frame, conf=0.25, classes=[0], verbose=False)

    if debug:
        cv2.namedWindow("YOLO26 Realtime Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO26 Realtime Detection", 320, 320)
        cv2.setWindowProperty("YOLO26 Realtime Detection", cv2.WND_PROP_TOPMOST, 1)

    print(f"准备开始检测，截取区域: {capture_region}")
    print("开始截图和推理，按 'q' 键退出...")

    mouse_ctrl = MouseController()
    prev_time = time.perf_counter()

    # 不断从捕获生成器里获取画面
    for frame in capture_screen_iter(region=capture_region):
        frame_start_time = time.perf_counter()

        fps = 1 / (frame_start_time - prev_time) if frame_start_time - prev_time > 0 else 0
        prev_time = frame_start_time

        # DXCam 已经直接产出 BGR 格式了，这里直接赋值，绕过 CPU 转换耗时
        frame_bgr = frame

        # YOLO 推理，设置 conf 过滤低置信度，classes=[0] 限制只检测人物，verbose=False 防止刷屏
        results = model.predict(source=frame_bgr, conf=0.3, classes=[0], verbose=False)

        # 提取目标中心点坐标并找到距离鼠标实际位置最近的目标
        mouse_x, mouse_y = mouse_ctrl.mouse_ctl.position

        closest_center = None
        min_distance = float("inf")

        for box in results[0].boxes:
            # xyxy 格式存放了 [x_min, y_min, x_max, y_max]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            # 将目标的局部坐标转换为屏幕绝对坐标
            abs_center_x = center_x + capture_region[0]
            abs_center_y = center_y + capture_region[1]

            # 计算到系统鼠标实际位置的距离平方
            distance = (abs_center_x - mouse_x) ** 2 + (abs_center_y - mouse_y) ** 2

            if distance < min_distance:
                min_distance = distance
                closest_center = (center_x, center_y)

        if closest_center:
            mouse_ctrl.update_target(closest_center[0], closest_center[1], capture_region[0], capture_region[1])
            print(f"🎯 最近目标的中心坐标: ({closest_center[0]:.1f}, {closest_center[1]:.1f})")
        else:
            mouse_ctrl.update_target(None, None, capture_region[0], capture_region[1])

        # 计算并输出当前帧从获取到推理结束的时间 (毫秒)
        inference_end_time = time.perf_counter()
        latency_ms = (inference_end_time - frame_start_time) * 1000

        if debug:
            # print(f"Frame Time: {latency_ms:.2f} ms")

            # 将带有边框和标签的检测图像渲染出来
            annotated_frame = results[0].plot()

            # 加上我们自己的 FPS 文本 (打在带边框的结果图像上)
            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(
                annotated_frame, f"Time: {latency_ms:.1f} ms", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # 显示最终画面
            cv2.imshow("YOLO26 Realtime Detection", annotated_frame)

            # 每帧渲染完毕后持续确保窗口置顶（防止部分系统自动取消置顶）
            cv2.setWindowProperty("YOLO26 Realtime Detection", cv2.WND_PROP_TOPMOST, 1)

            # 检测按键，如果按下 q 键则退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            if msvcrt.kbhit() and msvcrt.getch().lower() == b"q":
                break

    if debug:
        cv2.destroyAllWindows()
    mouse_ctrl.stop()


if __name__ == "__main__":
    run_detection((640, 220, 1280, 640))
