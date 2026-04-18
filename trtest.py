from ultralytics import YOLO

model = YOLO("yolo26n.pt") 

model.export(
    format="onnx",
    imgsz=640,           # 如果 640 跑不动再降到 320，640 是精度/速度平衡点
    half=False,           # 核心优化：GTX 1650 Ti 支持 FP16 导出，可显著减少显存占用
    simplify=True,       # 必须：简化计算图
    dynamic=False,       # 固定尺寸：利于加速器静态优化
    opset=12             # 稳定版本
)