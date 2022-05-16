# image-classification converter 

import time
import json
from json import load
import numpy as np
import os
import torch
import boto3


def timer(thunk, repeat=1, number=10, dryrun=3, min_repeat_ms=1000):
    """Helper function to time a function"""
    for i in range(dryrun):
        thunk()
    ret = []
    for _ in range(repeat):
        while True:
            beg = time.time()
            for _ in range(number):
                thunk()
            end = time.time()
            lat = (end - beg) * 1e3
            if lat >= min_repeat_ms:
                break
            number = int(max(min_repeat_ms / (lat / number) + 1, number * 1.618))
        ret.append(lat / number)
    return ret

def load_model(mtype ,bucket_name,model_name,batchsize):
    s3_client = boto3.client('s3') 
    
    if mtype == "base":
        os.makedirs(os.path.dirname(f'/tmp/base/{model_name}/'), exist_ok=True)
        s3_client.download_file(bucket_name, f'torch/base/{model_name}/model.pt', f'/tmp/base/{model_name}/model.pt')
        s3_client.download_file(bucket_name, f'torch/base/{model_name}/model_state_dict.pt', f'/tmp/base/{model_name}/model_state_dict.pt')
        
        PATH = f"/tmp/base/{model_name}/"
        model = torch.load(PATH+'model.pt')
        model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))

    elif mtype == "onnx":
        os.makedirs(os.path.dirname(f'/tmp/onnx/{model_name}/'), exist_ok=True)
        s3_client.download_file(bucket_name, f'torch/onnx/{model_name}/{model_name}_{batchsize}.onnx', f'/tmp/onnx/{model_name}/{model_name}_{batchsize}.onnx')
        
        # onnx 는 model path 필요 
        model = f"/tmp/onnx/{model_name}/{model_name}_{batchsize}.onnx"
        
    elif mtype == "tvm":
        os.makedirs(os.path.dirname(f'/tmp/tvm/{model_name}/'), exist_ok=True)
        s3_client.download_file(bucket_name, f'torch/tvm/arm/{model_name}/{model_name}_{batchsize}.tar', f'/tmp/tvm/{model_name}/{model_name}_{batchsize}.tar')
        
        model = f"/tmp/tvm/{model_name}/{model_name}_{batchsize}.tar"

    return model

def base_serving(bucket_name,model_name,batchsize,imgsize=224,repeat=10):
    # random data
    input_shape = (batchsize, 3, imgsize, imgsize)
    data_array = np.random.uniform(0, 255, size=input_shape).astype("float32")
    torch_data = torch.tensor(data_array)

    model= load_model("base",bucket_name,model_name,batchsize)
    model.eval()

    res = timer(lambda: model(torch_data),
                repeat=3,
                dryrun=5,
                min_repeat_ms=1000)

    print(f"Pytorch {model_name} latency for batch {batchsize} : {np.mean(res):.2f} ms")

    
def onnx_serving(bucket_name,model_name,batchsize,imgsize=224,repeat=10):
    import onnxruntime as ort

    model_path = load_model("onnx",bucket_name,model_name,batchsize)

    session = ort.InferenceSession(model_path)
    session.get_modelmeta()
    inname = [input.name for input in session.get_inputs()]
    outname = [output.name for output in session.get_outputs()]
    
    if model_name == "inception_v3":
        imgsize=299
    image_shape = (3, imgsize, imgsize)
    data_shape = (batchsize,) + image_shape
    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
        
    time_list = []
    for i in range(repeat):
        start_time = time.time()
        session.run(outname, {inname[0]: data})
        running_time = time.time() - start_time
        time_list.append(running_time)


    time_medium = np.median(np.array(time_list[1:]))
    print(f"{model_name} inference time median : {time_medium*1000} ms")


def tvm_serving(bucket_name,model_name,batchsize,imgsize=224):
    import tvm
    from tvm import relay
    import tvm.contrib.graph_executor as runtime

    input_name = "input0"
    if model_name == "inception_v3":
        imgsize==299
    input_shape = (batchsize, 3, imgsize, imgsize)
    output_shape = (batchsize, 1000)
    
    model_path = load_model("tvm",bucket_name,model_name,batchsize)
    loaded_lib = tvm.runtime.load_module(model_path)

    dev = tvm.cpu()
    module = runtime.GraphModule(loaded_lib["default"](dev))
    data = np.random.uniform(size=input_shape)
    module.set_input(input_name, data)
    ftimer = module.module.time_evaluator("run", dev, min_repeat_ms=500, repeat=3)
    prof_res = np.array(ftimer().results) * 1000
    print(f"TVM {model_name} latency for batch {batchsize} : {np.mean(prof_res):.2f} ms")
    

def lambda_handler(event, context):
    handler_start = time.time()
    #event = event['body-json']
    
    bucket_name = os.environ.get('BUCKET_NAME')
    model_name = event['model_name']
    inf_type=event['inference_type']
    batchsize = event['batchsize']

    
    start_time = time.time()
    if inf_type == "onnx":
        print("Torch model to ONNX model serving")
        onnx_serving(bucket_name,model_name,batchsize)

    elif inf_type == "tvm":
        tvm_serving(bucket_name,model_name,batchsize)

    elif inf_type == "base":
        base_serving(bucket_name,model_name,batchsize)
    running_time = time.time() - start_time
    return {'handler_time': running_time}



#test 
# bucket_name = ''
# model_name = 'mobilenet_v2'
# inf_type='base'
# batchsize = 1

    
# if inf_type == "onnx":
#     print("Torch model to ONNX model serving")
#     onnx_serving(bucket_name,model_name,batchsize)

# elif inf_type == "tvm":
#     tvm_serving(bucket_name,model_name,batchsize)

# elif inf_type == "base":
#     base_serving(bucket_name,model_name,batchsize)
