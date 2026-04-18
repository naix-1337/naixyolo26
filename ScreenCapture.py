import dxcam
import cv2
import time

def capture_screen_iter(region=None):
    """
    使用 DXCam 截取屏幕区域，并将画面作为迭代器逐帧产出
    """
    camera = dxcam.create(region=region)
    camera.start(target_fps=60, video_mode=True)
    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is not None:
                yield frame
    finally:
        camera.stop()
        del camera

def capture_screen(region=None):
    """
    使用 DXCam 截取屏幕区域
    :param region: 截取区域 (left, top, right, bottom)。如果不指定则截取全屏。
    :return:
    """
    # 初始化 dxcam。如果截取指定区域，创建时不限制，而在 grab 时限制
    # DXCam 在初始化时可以指定区域，但为了灵活，我们在类外提供 region 参数
    camera = dxcam.create(region=region)
    # 启动截图
    camera.start(target_fps=60, video_mode=True)
    
    cv2.namedWindow("DXCam Capture", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("DXCam Capture", cv2.WND_PROP_TOPMOST, 1)
    
    print("开始截图，按 'q' 键退出...")
    prev_time = time.time()
    try:
        while True:
            # 获取最新截图
            frame = camera.get_latest_frame()
            if frame is not None:
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if current_time - prev_time > 0 else 0
                prev_time = current_time

                # dxcam 截取的图像是 RGB 格式（如果是 OpenCV 处理，通常需要 BGR，这里 dxcam 支持设置 output_color="BGR" 但默认是 RGB）
                # 这里我们默认手动转换
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # 在图像上显示帧率 (文本文字类型为 GREEN)
                cv2.putText(frame_bgr, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("DXCam Capture", frame_bgr)
            
            # 检测按键，如果按下 q 键则退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        # 清理资源
        camera.stop()
        del camera
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 可以在这里做个简单的测试，比如只截取 1920x1080 屏幕的一半（左上角 960x540）
    test_region = (0, 0, 960, 540)
    capture_screen(region=test_region)
