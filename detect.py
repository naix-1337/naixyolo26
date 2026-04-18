import cv2
import time
from ultralytics import YOLO
from ScreenCapture import capture_screen_iter

def run_detection(capture_region):
    print("加载 YOLO 模型...")
    # 使用现有的 yolo26n.pt
    model = YOLO("yolo26n.pt")
    
    cv2.namedWindow("YOLO26 Realtime Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("YOLO26 Realtime Detection", cv2.WND_PROP_TOPMOST, 1)
    
    print(f"准备开始检测，截取区域: {capture_region}")
    print("开始截图和推理，按 'q' 键退出...")
    
    prev_time = time.time()
    
    # 不断从捕获生成器里获取画面
    for frame in capture_screen_iter(region=capture_region):
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time - prev_time > 0 else 0
        prev_time = current_time
        
        # 将画面转为 BGR (dxcam 捕获是 RGB，OpenCV 需要 BGR)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # YOLO 推理，设置 conf 可以过滤低置信度，verbose=False 不打印每帧结果防止刷屏
        results = model.predict(source=frame_bgr, conf=0.25, verbose=False)
        
        # 将带有边框和标签的检测图像渲染出来
        annotated_frame = results[0].plot()
        
        # 加上我们自己的 FPS 文本 (打在带边框的结果图像上)
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示最终画面
        cv2.imshow("YOLO26 Realtime Detection", annotated_frame)
        
        # 检测按键，如果按下 q 键则退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection((640, 220, 1280, 640))
