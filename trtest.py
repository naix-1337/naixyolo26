import tensorrt as trt
import os

# 1. 显式设置日志记录器
logger = trt.Logger(trt.Logger.INFO)


# 2. 这里的代码绕过 Ultralytics，直接使用 TensorRT 原生 API
def build_engine(onnx_file_path, engine_file_path):
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    config = builder.create_builder_config()
    # 针对 1650 Ti 的显存设置 (2GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    # 开启 FP16
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 构建引擎
    print("正在构建 Engine，这可能需要几分钟，请不要关闭窗口...")
    serialized_engine = builder.build_serialized_network(network, config)

    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"导出成功: {engine_file_path}")


# 执行
build_engine(r"D:\YOLO26\ultralytics\yolo26n_1650ti.onnx.onnx", r"D:\YOLO26\ultralytics\yolo26n_1650ti.engine")