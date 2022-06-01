import time
import json
from json import load
import numpy as np
import os
import boto3

BUCKET_NAME = os.environ.get('BUCKET_NAME')


def load_model(model_name, batchsize):
    s3_client = boto3.client('s3')

    import torch
    os.makedirs(os.path.dirname(f'/tmp/base/{model_name}/'), exist_ok=True)
    s3_client.download_file(BUCKET_NAME, f'models/torch/{model_name}/model.pt', f'/tmp/base/{model_name}/model.pt')
    s3_client.download_file(BUCKET_NAME, f'models/torch/{model_name}/model_state_dict.pt',
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

    time_list = []
    for i in range(repeat):
        start_time = time.time()
        model(torch_data)
        running_time = time.time() - start_time
        time_list.append(running_time)

    res = np.median(np.array(time_list[1:]))
    return res


def lambda_handler(event, context):
    start_time = time.time()
    
    for e in event:
        if e['execute'] == True:
            model_name = e['model_name']
            hardware = e['hardware']
            framework = e['framework']
            optimizer = e['optimizer']
            lambda_memory = e['lambda_memory']
            batchsize = e['batchsize']
            user_email = e['user_email']
            convert_time = e['convert_time']
            break
    
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
            'execute': true,
            'convert_time': convert_time,
            'inference_time': running_time
        }
    else:
        return {
            'execute': false
        }
