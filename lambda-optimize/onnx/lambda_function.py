# image-classification converter 

import time
from json import load
import os
import boto3

BUCKET_NAME = os.environ.get('BUCKET_NAME')
s3_client = boto3.client('s3') 


def load_model(model_name):
    import torch
    
    os.makedirs(os.path.dirname(f'/tmp/torch/{model_name}/'), exist_ok=True)
    s3_client.download_file(BUCKET_NAME, f'models/torch/{model_name}/model.pt', f'/tmp/torch/{model_name}/model.pt')
    s3_client.download_file(BUCKET_NAME, f'models/torch/{model_name}/model_state_dict.pt', f'/tmp/torch/{model_name}/model_state_dict.pt')
        
    PATH = f"/tmp/torch/{model_name}/"
    model = torch.load(PATH+'model.pt')
    model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))

    return model


def optimize_onnx(model,model_name,batchsize,imgsize=224,repeat=10):
    import torch.onnx
    import hashlib
    # 원본 모델 

    if model_name == "inception_v3":
        imgsize=299
    
    os.makedirs(os.path.dirname(f'/tmp/onnx/{model_name}/'), exist_ok=True)
    output_onnx = f'/tmp/onnx/{model_name}/{model_name}_{batchsize}.onnx'
    print("Exporting model to ONNX format at '{}'".format(output_onnx))

    convert_start_time = time.time()
    input_names = ["input0"]
    output_names = ["output0"]
    inputs = torch.randn(batchsize, 3, imgsize, imgsize)

    torch_out = torch.onnx._export(model, inputs, output_onnx, export_params=True, verbose=False,
                                input_names=input_names, output_names=output_names)
    
    convert_time = time.time()-convert_start_time
    print("Convert Complete")

    info = f'torch_{model_name}_{batchsize}'
    hinfo = hashlib.sha256(info.encode())
    
    #s3에 업로드 
    s3_client.upload_file(f'/tmp/onnx/{model_name}/{model_name}_{batchsize}.onnx',BUCKET_NAME,f'models/onnx/{hinfo.hexdigest()}.onnx')
    print("S3 upload done")

    return convert_time

def lambda_handler(event, context):    
    model_name = event['model_name']
    batchsize = event['batchsize']

    model = load_model(model_name)

    start_time = time.time()
    print("Model optimize - Torch model to ONNX model")
    convert_time = optimize_onnx(model,model_name,batchsize)

    running_time = time.time() - start_time
    return {'model':model_name,'batchsize':batchsize, 'framework':'torch','handler_time': running_time , 'convert_time':convert_time}
