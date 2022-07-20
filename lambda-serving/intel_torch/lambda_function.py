import time
import json
from json import load
import numpy as np
import os
import boto3

BUCKET_NAME = os.environ.get('BUCKET_NAME')


def load_model(model_name, model_size):
    s3_client = boto3.client('s3')

    import torch
    os.makedirs(os.path.dirname(f'/tmp/base/{model_name}_{model_size}/'), exist_ok=True)
    s3_client.download_file(BUCKET_NAME, f'models/torch/{model_name}_{model_size}/model.pt', f'/tmp/base/{model_name}_{model_size}/model.pt')

    PATH = f"/tmp/base/{model_name}_{model_size}/"
    model = torch.load(PATH + 'model.pt')

    return model


def base_serving(model_name, model_size, batchsize, imgsize=224, repeat=10):
    import torch
    # random data
    input_shape = (batchsize, 3, imgsize, imgsize)
    data_array = np.random.uniform(0, 255, size=input_shape).astype("float32")
    torch_data = torch.tensor(data_array)

    model = load_model(model_name, model_size)
    model.eval()

    time_list = []
    for i in range(repeat):
        start_time = time.time()
        model(torch_data)
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
    request_id = context.aws_request_id
    log_group_name = context.log_group_name

    if "base" in optimizer:
        start_time = time.time()
        res = base_serving(model_name, model_size, batchsize)
        running_time = time.time() - start_time

        return {
            'model_name': model_name,
            'model_size': model_size,
            'hardware': "intel",
            'framework': framework,
            'optimizer': "base",
            'lambda_memory': lambda_memory,
            'batchsize': batchsize,
            'user_email': user_email,
            'execute': True,
            'convert_time': 0,
            'request_id' : request_id,
            'log_group_name' : log_group_name
            'inference_time': res,
        }
    else:
        return {
            'execute': False
        }
