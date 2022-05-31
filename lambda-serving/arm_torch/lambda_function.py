# image-classification converter 

import time
import json
from json import load
import numpy as np
import os
import boto3

BUCKET_NAME = os.environ.get('BUCKET_NAME')


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

def load_model(model_name, batchsize):
    s3_client = boto3.client('s3')

    import torch
    os.makedirs(os.path.dirname(f'/tmp/base/{model_name}/'), exist_ok=True)
    s3_client.download_file(BUCKET_NAME, f'torch/base/{model_name}/model.pt', f'/tmp/base/{model_name}/model.pt')
    s3_client.download_file(BUCKET_NAME, f'torch/base/{model_name}/model_state_dict.pt',
                            f'/tmp/base/{model_name}/model_state_dict.pt')

    PATH = f"/tmp/base/{model_name}/"
    model = torch.load(PATH + 'model.pt')
    model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))

    return model


def base_serving(model_name, batchsize, imgsize=224, repeat=10):
    import torch
    # random data
    input_shape = (batchsize, 3, imgsize, imgsize)
    data_array = np.random.uniform(0, 255, size=input_shape).astype("float32")
    torch_data = torch.tensor(data_array)

    model = load_model(model_name, batchsize)
    model.eval()
    
    res = timer(lambda: model(torch_data),
                repeat=repeat,
                dryrun=5,
                min_repeat_ms=1000)
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

    if optimizer == "base":
        res = base_serving(model_name, batchsize)
        running_time = time.time() - start_time
        return {
            'model_name': model_name,
            'hardware': hardware,
            'framework': framework,
            'optimizer': optimizer,
            'lambda_memory': lambda_memory,
            'batchsize': batchsize,
            'user_email': user_email,
            'execute': false,
            'inference_time': running_time
        }
    else:
        return {
            'model_name': model_name,
            'hardware': hardware,
            'framework': framework,
            'optimizer': optimizer,
            'lambda_memory': lambda_memory,
            'batchsize': batchsize,
            'user_email': user_email,
            'execute': false
        }
