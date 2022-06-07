import time
import json
from json import load
import numpy as np
import os
import boto3

BUCKET_NAME = os.environ.get('BUCKET_NAME')


def load_model(model_name, model_size):
    s3_client = boto3.client('s3')

    os.makedirs(os.path.dirname(f'/tmp/onnx/'), exist_ok=True)
    s3_client.download_file(BUCKET_NAME, f'models/onnx/{model_name}_{model_size}.onnx',
                            f'/tmp/onnx/{model_name}_{model_size}.onnx')

    # onnx 는 model path 필요 
    model = f"/tmp/onnx/{model_name}_{model_size}.onnx"

    return model


def onnx_serving(model_name, model_size, batchsize, imgsize=224, repeat=10):
    import onnxruntime as ort

    model_path = load_model(model_name, model_size)

    session = ort.InferenceSession(model_path)
    session.get_modelmeta()
    inname = [input.name for input in session.get_inputs()]
    outname = [output.name for output in session.get_outputs()]

    if model_name == "inception_v3":
        imgsize = 299
    image_shape = (3, imgsize, imgsize)
    data_shape = (batchsize,) + image_shape
    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

    time_list = []
    for i in range(repeat):
        start_time = time.time()
        session.run(outname, {inname[0]: data})
        running_time = time.time() - start_time
        time_list.append(running_time)

    res = np.median(np.array(time_list[1:]))
    return res


def lambda_handler(event, context):
    start_time = time.time()

    model_name = event['model_name']
    model_size = event['model_size']
    hardware = event['hardware']
    framework = event['framework']
    optimizer = event['optimizer']
    lambda_memory = event['lambda_memory']
    batchsize = event['batchsize']
    user_email = event['user_email']
    convert_time = event['convert_time']

    if "onnx" in optimizer and "intel" in hardware:
        res = onnx_serving(model_name, model_size, batchsize)
        running_time = time.time() - start_time
        return {
            'model_name': model_name,
            'model_size': model_size,
            'hardware': "intel",
            'framework': framework,
            'optimizer': "onnx",
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
