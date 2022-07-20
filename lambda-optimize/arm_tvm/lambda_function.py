# image-classification converter 

import time
import numpy as np
import os
import boto3
import torch

BUCKET_NAME = os.environ.get('BUCKET_NAME')
s3_client = boto3.client('s3') 


def load_model(framework,model_name,model_size):    
    import onnx
    os.makedirs(os.path.dirname(f'/tmp/{framework}/{model_name}_{model_size}/'), exist_ok=True)
    PATH = f"/tmp/{framework}/{model_name}_{model_size}/"

    if "onnx" in framework :
        s3_client.download_file(BUCKET_NAME, f'models/{framework}/{model_name}_{model_size}.onnx', f'/tmp/{framework}/{model_name}_{model_size}/model.onnx')
        model = onnx.load(PATH+'model.onnx')
        framework="onnx"
    else:
        s3_client.download_file(BUCKET_NAME, f'models/{framework}/{model_name}_{model_size}/model.pt', f'/tmp/{framework}/{model_name}_{model_size}/model.pt')
        model = torch.load(PATH+'model.pt')
        framework="torch"
   
    return model

def optimize_tvm(framework,model,model_name,batchsize,model_size,imgsize=224,layout="NHWC"):
    import tvm
    from tvm import relay

    input_shape = (batchsize, 3, imgsize, imgsize)
    data_array = np.random.uniform(0, 255, size=input_shape).astype("float32")
   
    if "onnx" in framework:   
        shape_dict = {"input0": data_array.shape}
        mod, params = relay.frontend.from_onnx(model, shape=shape_dict)
    else:
        torch_data = torch.tensor(data_array)
        model.eval()
        traced_model = torch.jit.trace(model, torch_data)
        mod, params = relay.frontend.from_pytorch(traced_model, input_infos=[('input0', input_shape)],default_dtype="float32")
        
    if layout == "NHWC":
        desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
        seq = tvm.transform.Sequential(
            [
                relay.transform.RemoveUnusedFunctions(),
                relay.transform.ConvertLayout(desired_layouts),
            ]
        )
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
    else:
        assert layout == "NCHW"

    target = tvm.target.arm_cpu()
    # target = 'llvm -device=arm_cpu -mtriple=aarch64-linux-gnu'
    
    convert_start_time = time.time()
    with tvm.transform.PassContext(opt_level=3,required_pass=["FastMath"]):
        mod = relay.transform.InferType()(mod)
        lib = relay.build(mod, target=target, params=params)


    os.makedirs(os.path.dirname(f'/tmp/tvm/arm/{model_name}/'), exist_ok=True)
    lib.export_library(f"/tmp/tvm/arm/{model_name}/{model_name}_{batchsize}.tar")
    print("export done :",f"{model_name}_{batchsize}.tar")
    convert_time = time.time() - convert_start_time
    
    s3_client.upload_file(f'/tmp/tvm/arm/{model_name}/{model_name}_{batchsize}.tar',BUCKET_NAME,f'models/tvm/arm/{framework}/{model_name}_{model_size}.tar')
    print("S3 upload done")

    return convert_time

    
def lambda_handler(event, context):    
    model_name = event['model_name']
    model_size = event['model_size']
    hardware = "arm"
    framework = event['framework']
    optimizer = event['configuration'][hardware]
    batchsize = event['batchsize']
    user_email = event ['user_email']
    lambda_memory = event['lambda_memory']
    convert_time = 0
    
    if "tvm" in optimizer:
        start_time = time.time()
        model = load_model(framework,model_name,model_size)
        load_time = time.time() - start_time
        print("Model load time : ",load_time)

        print(f"Hardware optimize - {framework} model to TVM model")
        convert_time = optimize_tvm(framework,model,model_name,batchsize,model_size)

    return {
            'model_name': model_name,
            'model_size': model_size,
            'configuration': event['configuration'],
            'framework': framework,
            'lambda_memory': lambda_memory,
            'batchsize': batchsize,
            'user_email': user_email,
            'convert_time': convert_time
        }
