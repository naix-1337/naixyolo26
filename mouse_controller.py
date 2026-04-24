import importlib.util
import sys
import threading
import time

from pynput import mouse

# 动态加载存在特殊命名的 logitech.test.py 模块
spec = importlib.util.spec_from_file_location("logitech_test", "logitech.test.py")
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

        # 平滑系数。增加刷新频率后，可以适当调小系数获得丝滑且死锁的连续移动
        self.smooth = 0.4

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
        """传入最新最近目标的局部坐标及截图区域(capture_region)的左上角绝对坐标(offset_x, offset_y)."""
        if center_x is not None and center_y is not None:
            # 还原目标的屏幕绝对物理坐标
            abs_target_x = center_x + offset_x
            abs_target_y = center_y + offset_y

            # 获取当下鼠标在系统里的真实绝对位置
            mouse_x, mouse_y = self.mouse_ctl.position

            # 计算鼠标实际需要移动的两轴距离
            dx = abs_target_x - mouse_x
            dy = abs_target_y - mouse_y

            # 基于注释修正为：正数向右，负数向左，匹配标准的屏幕坐标体系
            self.rel_x = dx
            self.rel_y = dy
        else:
            self.rel_x = 0
            self.rel_y = 0

    def on_click(self, x, y, button, pressed):
        # x1 表示后退侧键
        if button == mouse.Button.x1:
            self.is_aiming = pressed

    def move_loop(self):
        while self.running:
            # 只要处于瞄准状态且还有剩余距离，就持续移动，突破原先的一帧移一次的限制
            if self.is_aiming and (self.rel_x != 0 or self.rel_y != 0):
                # 每次移动剩下相对距离的一部分
                move_x = int(self.rel_x * self.smooth)
                move_y = int(self.rel_y * self.smooth)

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
