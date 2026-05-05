# detect.py 多线程异步优化设计

## 目标

将 `detect.py` 从同步单线程改为多线程异步处理，提升帧率同时保持端到端延迟可控（平衡优化）。

## 当前问题

原始 pipeline 完全同步：

```
capture → model.predict() → find_target → render → imshow → (loop)
```

YOLO 推理（GPU）期间 CPU 空闲等待，截图无法并行；渲染也阻塞下一帧的获取。

## 方案：生产者-消费者 + 最新帧策略

### 架构

```
Capture 线程（生产者）              主线程（消费者：推理 + 渲染 + 鼠标）
 ┌──────────────────┐     ┌──────────────┐     ┌────────────────────────┐
 │ DXCam get_frame  │────→│  FrameBuffer │←────│ model.predict()       │
 │ 无等待持续抓帧    │     │ (thd-safe)   │     │ _find_closest_target  │
 │ 覆盖旧帧，不积压  │     │ latest frame │     │ _render_frame         │
 └──────────────────┘     │  Lock 保护    │     │ cv2.imshow            │
                           └──────────────┘     │ mouse_ctrl.update     │
                                                └────────────────────────┘
```

### FrameBuffer

线程安全单帧缓冲区：

- `set(frame)`: Capture 线程调用，无阻塞覆盖
- `get()`: 主线程调用，取走并清空 buffer（保证始终处理最新帧）

### Capture 线程

独立线程运行 `capture_worker()`：

- 在该线程内创建/销毁 DXCam 实例
- 持续 `get_latest_frame()` 写入 FrameBuffer
- 通过 `threading.Event` 控制退出

### 主线程

循环逻辑：

1. `buffer.get()` 取最新帧（无帧时 sleep 让出 CPU）
2. 计算 FPS
3. `model.predict()` 推理
4. `_find_closest_target()` 找目标
5. `mouse_ctrl.update_target()` 更新鼠标
6. 计算延迟
7. 调试模式：渲染 + `imshow`
8. 检测退出按键

### 退出流程

1. `stop_event.set()` 通知 Capture 线程退出
2. `capture_thread.join(timeout=2)` 等待子线程结束
3. `cv2.destroyAllWindows()` + `mouse_ctrl.stop()`

## 改动范围

| 文件 | 改动 |
|------|------|
| `detect.py` | 新增 `FrameBuffer` 类、`capture_worker()` 函数；重构 `run_detection()` 主循环 |
| 其他文件 | 无改动 |

## 与现有模块兼容性

- `ScreenCapture.capture_screen_iter` → 不再使用，替换为 `capture_worker`
- `MouseController` → 不变，主线程同步调用 `update_target`
- `_find_closest_target` / `_render_frame` / `_check_exit` → 不变

## 未涉及

- 不修改 ScreenCapture.py / mouse_controller.py
- 不引入外部依赖（仅使用 `threading` 标准库）
