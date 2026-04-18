from detect import run_detection

def main():
    # 屏幕分辨率是 1920x1080
    # 定义要截取的区域 (left, top, right, bottom)
    # 这里我们截取全屏的中心区域，宽 320，高 320
    # 左上角坐标: x = (1920 - 320) / 2 = 800
    # 左上角坐标: y = (1080 - 320) / 2 = 380
    # 右下角坐标: right = 800 + 320 = 1120
    # 右下角坐标: bottom = 380 + 320 = 700
    
    capture_region = (640, 220, 1280, 860)
    # capture_region = (800, 380, 1120, 700)
    
    DEBUG_MODE = True
    
    print(f"准备调用截屏并进行目标检测，截取区域: {capture_region}，调试模式: {DEBUG_MODE}")
    # 调用目标检测逻辑
    run_detection(capture_region, debug=DEBUG_MODE)

if __name__ == "__main__":
    main()
