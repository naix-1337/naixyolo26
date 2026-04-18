from ultralytics import YOLO

model = YOLO("yolo26n.pt")
# 导出为 TensorRT 格式，开启半精度
model.export(format="engine", device=0, half=False, simplify=True)