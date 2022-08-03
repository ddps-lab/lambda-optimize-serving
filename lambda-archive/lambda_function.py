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
    response = None
    max_memory_used = 0

    start_query_response = log_client.start_query(
        logGroupName=log_group_name,
        startTime=int((datetime.today() - timedelta(hours=24)).timestamp()),
        endTime=int((datetime.now() + timedelta(hours=24)).timestamp()),
        queryString=query,
    )
    query_id = start_query_response['queryId']
    while response == None or response['status'] == 'Running':
        time.sleep(1)
        response = log_client.get_query_results(
            queryId=query_id
        )

    res = response['results'][0]
    for r in res:
        if r['field'] == '@maxMemoryUsed':
            max_memory_used = int(float(r['value']) / 1000000)

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

def ses_send(user_email,info,max_memory_used):
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
                                "Data": f"AYCI convert time results\n---------------------------------------\n{info['model_name']} convert using {info['optimizer'].upper()} on {info['hardware'].upper()} \n{info['model_name']} size : {info['model_size']} MB\nConvert {info['model_name']} latency : {round(info['convert_time'],4)} s\n\nAYCI inference time results\n---------------------------------------\n{info['model_name']} inference Done!\n{info['model_name']} size : {info['model_size']} MB\nInference batchsize : {info['batchsize']}\nInference {info['model_name']} latency on {info['hardware'].upper()}: {round(info['inference_time'],4)} s\n-----------------------------------------------\nLambda memory size : {info['lambda_memory']}\nMax Memory Used : {max_memory_used}",
                                "Charset": "UTF-8"
                            },
                        }
                    }
    mfile_path = "/tmp/message.json"

    with open(mfile_path, 'w', encoding='utf-8') as mfile:
        json.dump(message_format, mfile)

    os.system("aws ses send-email --from allyoucaninference@gmail.com --destination=file:///tmp/destination.json --message=file:///tmp/message.json")

    
    
def lambda_handler(event, context):

    for i in range(len(event)):
        if event[i]['execute'] :
            user_email = event[i]['user_email']
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
            max_memory_used = getMemoryUsed(info)
            print(max_memory_used)
            info['max_memory_used'] = max_memory_used
            upload_data(info)
            ses_send(user_email,info,info['max_memory_used'])

        else:
            pass


    return { 'result':'upload done'}
