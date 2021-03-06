import time
import json
from json import load
import numpy as np
import os
import boto3

BUCKET_NAME = os.environ.get('BUCKET_NAME')

def load_model(framework,model_name, model_size):
    s3_client = boto3.client('s3')

    os.makedirs(os.path.dirname(f'/tmp/tvm/'), exist_ok=True)
    if "onnx" in framework:
        s3_client.download_file(BUCKET_NAME, f'models/tvm/intel/onnx/{model_name}_{model_size}.tar',
                                f'/tmp/tvm/{model_name}_{model_size}.tar')
    else:
        s3_client.download_file(BUCKET_NAME, f'models/tvm/intel/{model_name}_{model_size}.tar',
                                f'/tmp/tvm/{model_name}_{model_size}.tar')

    model = f"/tmp/tvm/{model_name}_{model_size}.tar"

    return model


def tvm_serving(framework,model_name, model_size, batchsize, imgsize=224, repeat=10):
    import tvm
    from tvm import relay
    import tvm.contrib.graph_executor as runtime

    input_name = "input0"
    if model_name == "inception_v3":
        imgsize == 299
    input_shape = (batchsize, 3, imgsize, imgsize)
    output_shape = (batchsize, 1000)

    model_path = load_model(framework,model_name, model_size)
    loaded_lib = tvm.runtime.load_module(model_path)

    target = "llvm -mcpu=core-avx2"
    dev = tvm.device(target, 0)
    module = runtime.GraphModule(loaded_lib["default"](dev))
    data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
    data = tvm.nd.array(data, dev)
    module.set_input(input_name, data)

    time_list = []
    for i in range(repeat):
        start_time = time.time()
        module.run(data=data)
        running_time = time.time() - start_time
        time_list.append(running_time)

    res = np.median(np.array(time_list[1:]))
    return res


def lambda_handler(event, context):
    model_name = event['model_name']
    model_size = event['model_size']
    hardware = "intel"
    framework = event['framework']
    optimizer = event['configuration'][hardware]
    lambda_memory = event['lambda_memory']
    batchsize = event['batchsize']
    user_email = event['user_email']
    convert_time = event['convert_time']
    request_id = context.aws_request_id
    log_group_name = context.log_group_name

    if "tvm" in optimizer:
        start_time = time.time()
        res = tvm_serving(framework,model_name, model_size, batchsize)
        running_time = time.time() - start_time

        return {
            'model_name': model_name,
            'model_size': model_size,
            'hardware': "intel",
            'framework': framework,
            'optimizer': "tvm",
            'lambda_memory': lambda_memory,
            'batchsize': batchsize,
            'user_email': user_email,
            'execute': True,
            'convert_time': convert_time,
            'inference_time': running_time,
            'request_id' : request_id,
            'log_group_name' : log_group_name
        }
    else:
        return {
            'execute': False
        }
