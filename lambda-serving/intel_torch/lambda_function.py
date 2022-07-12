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

def ses_send(user_email,info):
    dst_format = {"ToAddresses":[f"{user_email}"],
    "CcAddresses":[],
    "BccAddresses":[]}

    dfile_path = "/tmp/destination.json"

    with open(dfile_path, 'w', encoding='utf-8') as file:
        json.dump(dst_format, file)

    message_format = {
                        "Subject": {
                            "Data": "AYCI : AllYouCanInference results mail",
                            "Charset": "UTF-8"
                        },
                        "Body": {
                            "Text": {
                                "Data": f"AYCI inference time results\n---------------------------------------\n{info['model_name']} inference Done!\n{info['model_name']} size : {info['model_size']} MB\nLambda memory size : {info['lambda_memory']}\nInference batchsize : {info['batchsize']}\nInference {info['model_name']} latency on Intel: {round(info['inference_time'],4)} s",
                                "Charset": "UTF-8"
                            },
                        }
                    }

    mfile_path = "/tmp/message.json"

    with open(mfile_path, 'w', encoding='utf-8') as mfile:
        json.dump(message_format, mfile)

    os.system("aws ses send-email --from allyoucaninference@gmail.com --destination=file:///tmp/destination.json --message=file:///tmp/message.json")


def lambda_handler(event, context):

    model_name = event['model_name']
    model_size = event['model_size']
    hardware = event['hardware']
    framework = event['framework']
    optimizer = event['optimizer']
    lambda_memory = event['lambda_memory']
    batchsize = event['batchsize']
    user_email = event['user_email']
    request_id = context['aws_request_id']
    log_group_name = context['log_group_name']

    info = {
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
            'inference_time': 0
        }
        

    if "base" in optimizer and "intel" in hardware:
        start_time = time.time()
        res = base_serving(model_name, model_size, batchsize)
        running_time = time.time() - start_time
        info['inference_time']=running_time
        ses_send(user_email,info)

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
            'inference_time': running_time,
            'request_id' : request_id,
            'log_group_name' : log_group_name
        }
    else:
        return {
            'execute': False
        }
