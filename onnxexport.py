import torch
import onnx
from onnxsim import simplify

model = torch.load("yolo26n.pt", weights_only=False)
if type(model) is dict:
    model = model['model']  # Ultralytics 的 pt 文件实际是个字典，包含配套的信息，模型对象藏在 'model' 键里
model = model.eval().cuda().float()

# 1650 Ti 只有 4G 显存，必须锁死 Batch_Size = 1
# 如果 640x640 跑起来显存吃紧，甚至可以降级到 416x416 或 512x512
dummy_input = torch.randn(1, 3, 640, 640, device="cuda").float()

onnx_path = "yolo26n_1650ti.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=17,          
    do_constant_folding=True,  
    input_names=["images"],
    output_names=["output0"],
    # 绝对不要加 dynamic_axes，1650Ti 承受不起动态图的显存碎片
)

# 暴力化简
onnx_model = onnx.load(onnx_path)
model_simp, check = simplify(onnx_model)
assert check, "ONNX 化简失败"
onnx.save(model_simp, "yolo26s_1650ti_sim.onnx")