# image-classification & nlp converter 

import time
from json import load
import json
import numpy as np
import os
import boto3

BUCKET_NAME = os.environ.get('BUCKET_NAME')
s3_client = boto3.client('s3') 


def check_results(prefix,model_size,model_name):    
    exist=False

    obj_list = s3_client.list_objects(Bucket=BUCKET_NAME,Prefix=prefix)

    check = prefix + f'{model_name}_{model_size}.onnx'
    contents_list = obj_list['Contents']
    for content in contents_list:
        # print(content)
        if content['Key']== check : 
            exist=True
            break
    return exist 

def update_results(model_name,model_size,batchsize,convert_time,load_time):
    info = {'base_model_load_time':load_time,
            'convert_time' : convert_time}    

    with open(f'/tmp/{model_name}_{model_size}_{batchsize}_convert.json','w') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)  
    
    s3_client.upload_file(f'/tmp/{model_name}_{model_size}_{batchsize}_convert.json',BUCKET_NAME,f'results/onnx/{model_name}_{model_size}_convert.json')
    print("upload done : convert time results")



def load_model(model_name,model_size):
    import torch
    
    os.makedirs(os.path.dirname(f'/tmp/torch/{model_name}/'), exist_ok=True)
    s3_client.download_file(BUCKET_NAME, f'models/torch/{model_name}_{model_size}/model.pt', f'/tmp/torch/{model_name}/model.pt')
        
    PATH = f"/tmp/torch/{model_name}/"
    model = torch.load(PATH+'model.pt')

    return model


def optimize_onnx(wtype,model_name,batchsize,model_size,imgsize=224,seq_length=128):
    import torch.onnx
    import hashlib
    ######0. 원본 모델 
    start_time = time.time()
    model = load_model(model_name,model_size)
    load_time = time.time() - start_time
    
    #####1. onnx optimize 후 저장될 공간 만들기 
    os.makedirs(os.path.dirname(f'/tmp/onnx/{model_name}/'), exist_ok=True)
    output_onnx = f'/tmp/onnx/{model_name}/{model_name}_{batchsize}.onnx'
    print("Exporting model to ONNX format at '{}'".format(output_onnx))

    convert_start_time = time.time()
    input_names = ["input0"]
    output_names = ["output0"]

    if wtype == "img": 
        if model_name == "inception_v3":
            imgsize=299  
        inputs = torch.randn(batchsize, 3, imgsize, imgsize)

        torch.onnx.export(model, inputs, output_onnx, export_params=True, verbose=False,do_constant_folding=True,
                                input_names=input_names, output_names=output_names,dynamic_axes= {'input0' : {0 : 'batch_size'},    # variable length axes
                                'output0' : {0 : 'batch_size'}})

    elif wtype=="nlp":
        inputs = np.random.randint(0, 2000, size=(seq_length))
        token_types = np.random.randint(0,2,size=(seq_length))

        tokens_tensor = torch.tensor(np.array([inputs]))
        segments_tensors = torch.tensor(np.array([token_types]))

        torch.onnx.export(model,(tokens_tensor,segments_tensors), output_onnx, export_params=True, verbose=False,do_constant_folding=True,
                                input_names=input_names, output_names=output_names,dynamic_axes= {'input0' : {0 : 'batch_size'},    # variable length axes
                                'output0' : {0 : 'batch_size'}})


    convert_time = time.time()-convert_start_time
    print("Convert Complete")


    #s3에 업로드 
    s3_client.upload_file(f'/tmp/onnx/{model_name}/{model_name}_{batchsize}.onnx',BUCKET_NAME,f'models/onnx/{model_name}_{model_size}.onnx')
    print("S3 upload done")

    return load_time,convert_time


def lambda_handler(event, context):    
    workload_type = event['workload_type']
    model_name = event['model_name']
    model_size = event['model_size']
    framework = event['framework']
    configuration = event['configuration']
    batchsize = event['batchsize']
    user_email = event ['user_email']
    lambda_memory = event['lambda_memory']

    # convert한 모델이 있는지 확인 
    prefix = 'models/onnx/'
    exist = check_results(prefix,model_size,model_name)

    if exist==False:
        if "onnx" in configuration["intel"] or "onnx" in configuration["arm"]:
            load_time, convert_time = optimize_onnx(workload_type,model_name,batchsize,model_size)
            update_results(model_name,model_size,batchsize,convert_time,load_time)

    return {
            'workload_type':workload_type,
            'model_name': model_name,
            'model_size': model_size,
            'configuration': configuration,
            'framework': framework,
            'lambda_memory': lambda_memory,
            'batchsize': batchsize,
            'user_email': user_email,
        }
