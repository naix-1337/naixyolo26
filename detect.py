import cv2
import time
import numpy as np
import msvcrt
from ultralytics import YOLO
from ScreenCapture import capture_screen_iter
from mouse_controller import MouseController

def run_detection(capture_region, debug=True):
    print("加载 YOLO 模型...")
    # 使用现有的模型
    model = YOLO("mrzh759s.onnx")
    
    # 预热推理
    print("模型预热中，执行 10 次空数据推理...")
    w = capture_region[2] - capture_region[0]
    h = capture_region[3] - capture_region[1]
    dummy_frame = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(10):
        model.predict(source=dummy_frame, conf=0.25, classes=[0, 1, 2], verbose=False)
    
    if debug:
        cv2.namedWindow("YOLO26 Realtime Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO26 Realtime Detection", 320, 320)
        cv2.setWindowProperty("YOLO26 Realtime Detection", cv2.WND_PROP_TOPMOST, 1)
    
    print(f"准备开始检测，截取区域: {capture_region}")
    print("开始截图和推理，按 'q' 键退出...")
    
    mouse_ctrl = MouseController()
    prev_time = time.perf_counter()
    frame_count = 0
    
    for frame in capture_screen_iter(region=capture_region):
        frame_start_time = time.perf_counter()
        frame_count += 1
        
        fps = 1 / (frame_start_time - prev_time) if frame_start_time - prev_time > 0 else 0
        prev_time = frame_start_time
        
        results = model.predict(source=frame, conf=0.3, classes=[0, 1, 2], verbose=False)
        
        # 向量化计算最近目标
        mouse_x, mouse_y = mouse_ctrl.mouse_ctl.position
        
        boxes = results[0].boxes
        closest_center = None
        xyxy = None
        
        if len(boxes) > 0:
            xyxy = boxes.xyxy
            cls_ids = boxes.cls
            centers = (xyxy[:, :2] + xyxy[:, 2:]) / 2.0
            
            centers_by_cls = {}
            for cls_id in [0, 1, 2]:
                mask = cls_ids == cls_id
                if mask.sum() > 0:
                    cls_centers = centers[mask]
                    centers_by_cls[cls_id] = cls_centers
                    print(f"📦 类别 {cls_id} 目标数: {len(cls_centers)}, 中心: {[(f'{c[0]:.1f}', f'{c[1]:.1f}') for c in cls_centers]}")
                else:
                    centers_by_cls[cls_id] = None
            
            # 只计算距离类型 2 目标的最近距离
            if centers_by_cls[2] is not None:
                cls2_centers = centers_by_cls[2]
                abs_centers_x = cls2_centers[:, 0] + capture_region[0]
                abs_centers_y = cls2_centers[:, 1] + capture_region[1]
                
                dx = abs_centers_x - mouse_x
                dy = abs_centers_y - mouse_y
                dists = dx.pow(2) + dy.pow(2)
                
                best_idx = dists.argmin().item()
                closest_center = (cls2_centers[best_idx, 0].item(), cls2_centers[best_idx, 1].item())
            
        if closest_center:
            mouse_ctrl.update_target(closest_center[0], closest_center[1], capture_region[0], capture_region[1])
            print(f"🎯 最近类型2目标的中心坐标: ({closest_center[0]:.1f}, {closest_center[1]:.1f})")
        else:
            mouse_ctrl.update_target(None, None, capture_region[0], capture_region[1])
        
        # 计算并输出当前帧从获取到推理结束的时间 (毫秒)
        inference_end_time = time.perf_counter()
        latency_ms = (inference_end_time - frame_start_time) * 1000
        
        if debug:
            annotated_frame = frame.copy()
            if xyxy is not None:
                xyxy_np = xyxy.cpu().numpy().astype(int)
                for (x1, y1, x2, y2) in xyxy_np:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Time: {latency_ms:.1f} ms", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("YOLO26 Realtime Detection", annotated_frame)
            
            if frame_count % 30 == 0:
                cv2.setWindowProperty("YOLO26 Realtime Detection", cv2.WND_PROP_TOPMOST, 1)
            
            # 检测按键，如果按下 q 键则退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if msvcrt.kbhit() and msvcrt.getch().lower() == b'q':
                break
            
    if debug:
        cv2.destroyAllWindows()
    mouse_ctrl.stop()

if __name__ == "__main__":
    run_detection((640, 220, 1280, 640))
