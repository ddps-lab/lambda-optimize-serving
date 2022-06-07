# inference results archived moddule 
import json
import os
import boto3

BUCKET_NAME = os.environ.get('BUCKET_NAME')
s3_client = boto3.client('s3') 


def upload_data(info):    
    model_name = info['model_name']
    model_size = info['model_size']
    optimizer = info['optimizer']
    batchsize = info['batchsize']
    lambda_memory = info['lambda_memory']
    with open(f'./{model_name}_{model_size}_{batchsize}_{lambda_memory}.json','w') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)  
    s3_client.upload_file(f'./{model_name}_{model_size}_{batchsize}_{lambda_memory}.json',BUCKET_NAME,f'results/{optimizer}/{model_name}_{model_size}_{batchsize}_{lambda_memory}.json')

def lambda_handler(event, context):    
    for i in range(len(event)):
        if event[i]['execute'] : 
            info = {
                'model_name':event[i]['model_name'],
                'model_size':event[i]['model_size'],
                'hardware':event[i]['hardware'],
                'framework':event[i]['framework'],
                'optimizer':event[i]['optimizer'],
                'lambda_memory':event[i]['lambda_memory'],
                'batchsize':event[i]['batchsize'],
                'convert_time':event[i]['convert_time'],
                'infernece_time':event[i]['inference_time']
            }
            upload_data(info)
        else:
            pass


    return {"upload inference results done"}

