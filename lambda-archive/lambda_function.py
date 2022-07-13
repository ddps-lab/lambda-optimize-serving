# inference results archived module
import json
import os
import boto3
from datetime import datetime, timedelta
import time

BUCKET_NAME = os.environ.get('BUCKET_NAME')
s3_client = boto3.client('s3') 
log_client = boto3.client('logs')

def getMemoryUsed(info):
    request_id = info['request_id']
    log_group_name = info['log_group_name']

    query = f"fields @maxMemoryUsed | sort @timestamp desc | filter @requestId='{request_id}' | filter @maxMemoryUsed like ''"
    start_query_response = log_client.start_query(
        logGroupName=log_group_name,
        startTime=int((datetime.today() - timedelta(hours=1)).timestamp()),
        endTime=int(datetime.now().timestamp()),
        queryString=query,
    )
    query_id = start_query_response['queryId']

    response = None
    while response == None or start_query_response['status'] == 'Running':
        time.sleep(1)
        response = log_client.get_query_results(
            queryId=query_id
        )

    max_memory_used = 0
    res = response['results'][0]
    for r in res:
        if r['field'] == '@maxMemoryUsed':
            max_memory_used = int(r['value']) / 1000000

    return max_memory_used

def upload_data(info):    
    model_name = info['model_name']
    model_size = info['model_size']
    optimizer = info['optimizer']
    batchsize = info['batchsize']
    lambda_memory = info['lambda_memory']
    hardware = info['hardware']
    with open(f'/tmp/{model_name}_{model_size}_{batchsize}_{lambda_memory}.json','w') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)  
    s3_client.upload_file(f'/tmp/{model_name}_{model_size}_{batchsize}_{lambda_memory}.json',BUCKET_NAME,f'results/{optimizer}/{hardware}/{model_name}_{model_size}_{batchsize}_{lambda_memory}.json')

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
                'inference_time':event[i]['inference_time'],
                'request_id':event[i]['request_id'],
                'log_group_name':event[i]['log_group_name']
            }
            upload_data(info)
            max_memory_used = getMemoryUsed(info)
            print(max_memory_used)
        else:
            pass


    return { 'result':'upload done'}
