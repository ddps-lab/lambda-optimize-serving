import time
import json
from json import load
import numpy as np
import os
import boto3
import tvm
from tvm import relay
import tvm.contrib.graph_executor as runtime

BUCKET_NAME = os.environ.get('BUCKET_NAME')
s3_client = boto3.client('s3')

def load_model(framework, model_name, model_size,batchsize):
    load_start = time.time()
    os.makedirs(os.path.dirname(f'/tmp/tvm/'), exist_ok=True)
    if "onnx" in framework:
        s3_client.download_file(BUCKET_NAME, f'models/tvm/arm/onnx/{model_name}_{model_size}_{batchsize}.tar',
                                f'/tmp/tvm/{model_name}_{model_size}.tar')
    else:
        s3_client.download_file(BUCKET_NAME, f'models/tvm/arm/{model_name}_{model_size}_{batchsize}.tar',
                                f'/tmp/tvm/{model_name}_{model_size}.tar')

    model = f"/tmp/tvm/{model_name}_{model_size}.tar"
    print(time.time() - load_start)
    return model

def update_results(framework,model_name,model_size,batchsize,lambda_memory,inference_mean, inference_median,inf_time_list,running_time):
    info = {
            'inference_mean' : inference_mean,
            'inference_median' : inference_median ,
            'inference_all' : inf_time_list,
            'inference_handler_time' : running_time
    }

    with open(f'/tmp/{model_name}_{model_size}_{batchsize}_{lambda_memory}_inference.json','w') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)  
    
    if "onnx" in framework :
        s3_client.upload_file(f'/tmp/{model_name}_{model_size}_{batchsize}_{lambda_memory}_inference.json',BUCKET_NAME,f'results/tvm/arm/onnx/{model_name}_{model_size}_{batchsize}_{lambda_memory}_inference.json')
        print("upload done : convert time results")
    else:
        s3_client.upload_file(f'/tmp/{model_name}_{model_size}_{batchsize}_{lambda_memory}_inference.json',BUCKET_NAME,f'results/tvm/arm/{model_name}_{model_size}_{batchsize}_{lambda_memory}_inference.json')
        print("upload done : convert time results")      


def tvm_serving(wtype, framework, model_name, model_size, batchsize, imgsize=224, repeat=10):
    dev = tvm.cpu()
    model_path = load_model(framework, model_name, model_size,batchsize)
    loaded_lib = tvm.runtime.load_module(model_path)
    module = runtime.GraphModule(loaded_lib["default"](dev))

    if wtype == 'img':
        if model_name == "inception_v3":
            imgsize = 299
        input_shape = (batchsize, 3, imgsize, imgsize)
        output_shape = (batchsize, 1000)
        input_name = "input0"

        # target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"
        # dev = tvm.device(target, 0)
        data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        data = tvm.nd.array(data, dev)
        module.set_input(input_name, data)

    elif wtype == 'nlp':
        dtype = "float32"
        seq_length = 128
        inputs = np.random.randint(0, 2000, size=(batchsize, seq_length)).astype(dtype)
        token_types = np.random.uniform(size=(batchsize, seq_length)).astype(dtype)
        dtype = 'float32'
        valid_length = np.asarray([seq_length] * batchsize).astype(dtype)

        data = tvm.nd.array(inputs, dev)
        token_types_nd = tvm.nd.array(token_types, dev)
        valid_length_nd = tvm.nd.array(valid_length, dev)
        module.set_input(data0=data, data1=token_types_nd, data2=valid_length_nd)

    time_list = []
    for i in range(repeat):
        start_time = time.time()
        module.run(data=data)
        running_time = time.time() - start_time
        time_list.append(running_time)

    median = np.median(np.array(time_list[1:]))
    mean = np.mean(np.array(time_list[1:]))

    return mean, median , time_list


def lambda_handler(event, context):
    workload_type = event['workload_type']
    model_name = event['model_name']
    model_size = event['model_size']
    hardware = "arm"
    framework = event['framework']
    optimizer = event['configuration'][hardware]
    lambda_memory = event['lambda_memory']
    batchsize = event['batchsize']
    user_email = event['user_email']
    request_id = context.aws_request_id
    log_group_name = context.log_group_name
    

    if "tvm" in optimizer:
        start_time = time.time()
        inference_mean, inference_median, inf_time_list = tvm_serving(workload_type, framework, model_name, model_size, batchsize)
        running_time = time.time() - start_time
        update_results(framework,model_name,model_size,batchsize,lambda_memory,inference_mean, inference_median,inf_time_list,running_time)

        return {
            'workload_type': workload_type,
            'model_name': model_name,
            'model_size': model_size,
            'hardware': "arm",
            'framework': framework,
            'optimizer': "tvm",
            'lambda_memory': lambda_memory,
            'batchsize': batchsize,
            'user_email': user_email,
            'execute': True,
            'request_id': request_id,
            'log_group_name': log_group_name
        }

    else:
        return {
            'execute': False
        }
