# image-classification converter 

import time
import json
from json import load
import numpy as np
import os
import boto3

BUCKET_NAME = os.environ.get('BUCKET_NAME')
s3_client = boto3.client('s3') 


def load_model(model_name):
    import torch
    
    os.makedirs(os.path.dirname(f'/tmp/base/{model_name}/'), exist_ok=True)
    s3_client.download_file(BUCKET_NAME, f'torch/base/{model_name}/model.pt', f'/tmp/base/{model_name}/model.pt')
    s3_client.download_file(BUCKET_NAME, f'torch/base/{model_name}/model_state_dict.pt', f'/tmp/base/{model_name}/model_state_dict.pt')
        
    PATH = f"/tmp/base/{model_name}/"
    model = torch.load(PATH+'model.pt')
    model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))

    return model


def optimize_onnx(model,model_name,batchsize,imgsize=224,repeat=10):
    import torch.onnx
    # 원본 모델 

    if model_name == "inception_v3":
        imgsize=299
    
    os.makedirs(os.path.dirname(f'/tmp/onnx/{model_name}/'), exist_ok=True)
    output_onnx = f'/tmp/onnx/{model_name}/{model_name}_{batchsize}.onnx'
    print("Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input0"]
    output_names = ["output0"]
    inputs = torch.randn(batchsize, 3, imgsize, imgsize)

    torch_out = torch.onnx._export(model, inputs, output_onnx, export_params=True, verbose=False,
                                input_names=input_names, output_names=output_names)
    
    #s3에 업로드 
    s3_client.upload_file(f'/tmp/onnx/{model_name}/{model_name}_{batchsize}.onnx',BUCKET_NAME,f'torch/onnx/{model_name}/{model_name}_{batchsize}.onnx')
    print("S3 upload done")

def optimize_tvm(model,model_name,batchsize,target,imgsize=224,layout="NHWC"):
    import tvm
    from tvm import relay
    import tvm.contrib.graph_executor as runtime
    import torch

    input_name = "input0"
    input_shape = (batchsize, 3, imgsize, imgsize)

    # data = np.random.uniform(size=input_shape)
    data_array = np.random.uniform(0, 255, size=input_shape).astype("float32")
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

    if target == "arm":
        target = tvm.target.arm_cpu()

    with tvm.transform.PassContext(opt_level=3,required_pass=["FastMath"]):
        mod = relay.transform.InferType()(mod)
        lib = relay.build(mod, target=target, params=params)

    if target=="arm":
        ft = "arm"
    else:
        ft = "intel"  

    os.makedirs(os.path.dirname(f'/tmp/tvm/{ft}/{model_name}/'), exist_ok=True)
    lib.export_library(f"/tmp/tvm/{ft}/{model_name}/{model_name}_{batchsize}.tar")
    print("export done :",f"{model_name}_{batchsize}.tar")

    s3_client.upload_file(f'/tmp/tvm/{ft}/{model_name}/{model_name}_{batchsize}.tar',BUCKET_NAME,f'torch/tvm/{ft}/{model_name}/{model_name}_{batchsize}.tar')
    print("S3 upload done")

    

def lambda_handler(event, context):
    handler_start = time.time()
    #event = event['body-json']
    
    model_name = event['model_name']
    convert_type=event['convert_type']
    batchsize = event['batchsize']
    target = event['target']


    model = load_model(model_name)

    start_time = time.time()
    if convert_type == "onnx":
        print("Model optimize - Torch model to ONNX model")
        optimize_onnx(model,model_name,batchsize)

    elif convert_type == "tvm":
        print("Hardware optimize - Torch model to TVM model")
        optimize_tvm(model,model_name,batchsize,target)


    running_time = time.time() - start_time
    return {'handler_time': running_time}



#test 
# BUCKET_NAME = 'dl-converted-models'
# model_name = 'mobilenet_v2'
# convert_type='onnx'
# batchsize = 1

# model = load_model(model_name)
# target = "arm"

# if convert_type == "onnx":
#     print("Model optimize - Torch model to ONNX model")
#     optimize_onnx(model,model_name,batchsize)

# elif convert_type == "tvm":
#     print("Hardware optimize - Torch model to TVM model")
#     optimize_tvm(model,model_name,batchsize,target)
