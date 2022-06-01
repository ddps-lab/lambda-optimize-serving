import time
import json
from json import load
import numpy as np
import os
import boto3

BUCKET_NAME = os.environ.get('BUCKET_NAME')


def load_model(model_name, batchsize):
    s3_client = boto3.client('s3')

    os.makedirs(os.path.dirname(f'/tmp/tvm/{model_name}/'), exist_ok=True)
    s3_client.download_file(BUCKET_NAME, f'models/tvm/arm/{model_name}/{model_name}_{batchsize}.tar',
                            f'/tmp/tvm/{model_name}/{model_name}_{batchsize}.tar')

    model = f"/tmp/tvm/{model_name}/{model_name}_{batchsize}.tar"

    return model


def tvm_serving(model_name, batchsize, imgsize=224, repeat=10):
    import tvm
    from tvm import relay
    import tvm.contrib.graph_executor as runtime

    input_name = "input0"
    if model_name == "inception_v3":
        imgsize == 299
    input_shape = (batchsize, 3, imgsize, imgsize)
    output_shape = (batchsize, 1000)

    model_path = load_model(model_name, batchsize)
    loaded_lib = tvm.runtime.load_module(model_path)

    dev = tvm.cpu()
    module = runtime.GraphModule(loaded_lib["default"](dev))
    data = np.random.uniform(size=input_shape)
    module.set_input(input_name, data)
    ftimer = module.module.time_evaluator("run", dev, min_repeat_ms=500, repeat=repeat)
    res = np.array(ftimer().results) * 1000
    return res


def lambda_handler(event, context):
    start_time = time.time()
    model_name = event['model_name']
    hardware = event['hardware']
    framework = event['framework']
    optimizer = event['optimizer']
    lambda_memory = event['lambda_memory']
    batchsize = event['batchsize']
    user_email = event['user_email']
    convert_time = event['convert_time']

    if optimizer == "tvm" and hardware == "arm":
        res = tvm_serving(model_name, batchsize)
        running_time = time.time() - start_time
        return {
            'model_name': model_name,
            'hardware': hardware,
            'framework': framework,
            'optimizer': optimizer,
            'lambda_memory': lambda_memory,
            'batchsize': batchsize,
            'user_email': user_email,
            'execute': True,
            'convert_time': convert_time,
            'inference_time': running_time
        }
    else:
        return {
            'execute': False
        }
