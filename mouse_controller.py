import os
import time
import threading
import importlib.util
import sys
from pynput import mouse
from pid import PIDController

# 动态加载存在特殊命名的 logitech.test.py 模块
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))
logitech_path = os.path.join(base_path, "logitech.test.py")
spec = importlib.util.spec_from_file_location("logitech_test", logitech_path)
logitech_test = importlib.util.module_from_spec(spec)
sys.modules["logitech_test"] = logitech_test
spec.loader.exec_module(logitech_test)

Logitech = logitech_test.Logitech

class MouseController:
    def __init__(self):
        self.rel_x = 0
        self.rel_y = 0
        
        self.is_aiming = False
        self.running = True
        self._lock = threading.Lock()
        
        self.pid = PIDController()
        self._had_target = False
        
        # 用于获取系统实际鼠标绝对位置的控制器
        self.mouse_ctl = mouse.Controller()
        
        # 监听鼠标侧键 (Button.x1 或 Button.x2 对应后侧键/前侧键)
        self.listener = mouse.Listener(on_click=self.on_click)
        self.listener.start()
        
        # 启动鼠标移动线程
        self.thread = threading.Thread(target=self.move_loop)
        self.thread.daemon = True
        self.thread.start()

    def update_target(self, center_x, center_y, offset_x, offset_y):
        """传入最新最近目标的局部坐标及截图区域(capture_region)的左上角绝对坐标(offset_x, offset_y)"""
        with self._lock:
            had_target = self._had_target
            if center_x is not None and center_y is not None:
                abs_target_x = center_x + offset_x
                abs_target_y = center_y + offset_y

                mouse_x, mouse_y = self.mouse_ctl.position

                dx = abs_target_x - mouse_x
                dy = abs_target_y - mouse_y

                self.rel_x = dx
                self.rel_y = dy

                if not had_target:
                    self.pid.reset()
                self._had_target = True
            else:
                self.rel_x = 0
                self.rel_y = 0

                if had_target:
                    self.pid.reset()
                self._had_target = False

    def on_click(self, x, y, button, pressed):
        # x1 表示后退侧键
        if button == mouse.Button.x1: 
            self.is_aiming = pressed

    def move_loop(self):
        while self.running:
            # 只要处于瞄准状态且还有剩余距离，就持续移动，突破原先的一帧移一次的限制
            if self.is_aiming:
                with self._lock:
                    if self.rel_x == 0 and self.rel_y == 0:
                        time.sleep(0.001)
                        continue
                    move_x, move_y = self.pid.update(self.rel_x, self.rel_y)
                    move_x = int(move_x)
                    move_y = int(move_y)
                    
                    # 如果剩余距离太小算出来的整数是0，但实际还有距离，则直接一次性移完
                    if move_x == 0 and self.rel_x != 0:
                        move_x = int(self.rel_x)
                    if move_y == 0 and self.rel_y != 0:
                        move_y = int(self.rel_y)
                    
                    if move_x != 0 or move_y != 0:
                        Logitech.mouse.move(move_x, move_y)
                        
                    # 扣除这次已经移动的距离
                    self.rel_x -= move_x
                    self.rel_y -= move_y
                    
            # 缩短休眠至1毫秒，大幅提高鼠标移动的执行频率（1000Hz）
            time.sleep(0.001)

    def stop(self):
        self.running = False
        self.listener.stop()
