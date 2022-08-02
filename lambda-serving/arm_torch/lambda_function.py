import time
import json
from json import load
import numpy as np
import os
import boto3
import hashlib
import torch

BUCKET_NAME = os.environ.get('BUCKET_NAME')


def load_model(model_name, model_size):
    s3_client = boto3.client('s3')

    os.makedirs(os.path.dirname(f'/tmp/base/{model_name}_{model_size}/'), exist_ok=True)
    s3_client.download_file(BUCKET_NAME, f'models/torch/{model_name}_{model_size}/model.pt',
                            f'/tmp/base/{model_name}_{model_size}/model.pt')

    PATH = f"/tmp/base/{model_name}_{model_size}/"
    model = torch.load(PATH + 'model.pt')

    return model


def base_serving(wtype, model_name, model_size, batchsize, imgsize=224, repeat=10):
    model = load_model(model_name, model_size)
    model.eval()

    time_list = []
    if wtype == 'img':
        if model_name == "inception_v3":
            imgsize = 299
        input_shape = (batchsize, 3, imgsize, imgsize)
        data_array = np.random.uniform(0, 255, size=input_shape).astype("float32")
        torch_data = torch.tensor(data_array)
        for i in range(repeat):
            start_time = time.time()
            model(torch_data)
            running_time = time.time() - start_time
            time_list.append(running_time)

    elif wtype == 'nlp':
        model.hybridize(static_alloc=True)
        seq_length = 128
        dtype = "float32"
        inputs = np.random.randint(0, 2000, size=(batchsize, seq_length)).astype(dtype)
        token_types = np.random.uniform(size=(batchsize, seq_length)).astype(dtype)
        valid_length = np.asarray([seq_length] * batchsize).astype(dtype)
        for i in range(repeat):
            start_time = time.time()
            model(inputs, token_types, valid_length)
            running_time = time.time() - start_time
            time_list.append(running_time)

    res = np.median(np.array(time_list[1:]))
    return res


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

    if "base" in optimizer:

        start_time = time.time()
        res = base_serving(workload_type, model_name, model_size, batchsize)
        running_time = time.time() - start_time

        return {
            'workload_type': workload_type,
            'model_name': model_name,
            'model_size': model_size,
            'hardware': "arm",
            'framework': framework,
            'optimizer': "base",
            'lambda_memory': lambda_memory,
            'batchsize': batchsize,
            'user_email': user_email,
            'execute': True,
            'convert_time': 0,
            'inference_time': running_time,
            'request_id': request_id,
            'log_group_name': log_group_name

        }
    else:
        return {
            'execute': False
        }
