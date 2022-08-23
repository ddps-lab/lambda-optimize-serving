# inference results archived module
import json
import os
import boto3
from datetime import datetime, timedelta
import time

BUCKET_NAME = os.environ.get('BUCKET_NAME')
s3_client = boto3.client('s3') 
log_client = boto3.client('logs')
s3 = boto3.resource('s3') 

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

def getLatency(prefix, check , gettype):
    obj_list = s3_client.list_objects(Bucket=BUCKET_NAME,Prefix=prefix)
    contents_list = obj_list['Contents']

    for content in contents_list:
        # print(content)
        if content['Key']== check : 
            # 파일 내용을 읽어오기
            obj = s3.Object(BUCKET_NAME,f"{check}")
            bytes_value = obj.get()['Body'].read()
            filejson = bytes_value.decode('utf8')
            fileobj = json.loads(filejson)
            print(fileobj)
            get_latency = fileobj[gettype]

    return get_latency

def upload_data(info,max_memory_used):        
    # get convert_time 
    try:
        if info['optimizer'] == "onnx":
            convert_prefix = f'results/{info["optimizer"]}/'
        else:
            convert_prefix = f'results/{info["optimizer"]}/{info["hardware"]}/'
        convert_check = convert_prefix + f'{info["model_name"]}_{info["model_size"]}_{info["batchsize"]}_convert.json'
        convert_time = getLatency(convert_prefix, convert_check, "convert_time")
    except:
        # base 인 경우 convert time 0 
        convert_time = 0
    # get inference_time 
    prefix = f'results/{info["optimizer"]}/{info["hardware"]}/'
    inference_check = prefix + f'{info["model_name"]}_{info["model_size"]}_{info["batchsize"]}_{info["lambda_memory"]}_inference.json'
    inference_time = getLatency(prefix,inference_check,"inference_median")


    get_info = {
            'model_name':info['model_name'],
            'model_size':info['model_size'],
            'hardware':info['hardware'],
            'framework':info['framework'],
            'optimizer':info['optimizer'],
            'lambda_memory':info['lambda_memory'],
            'batchsize':info['batchsize'],
            'convert_time':convert_time,
            'inference_time':inference_time,
            'user_email':info['user_email'],
            'request_id':info['request_id'],
            'log_group_name':info['log_group_name'],
            'max_memory_used':max_memory_used
        }


    with open(f'/tmp/{get_info["model_name"]}_{get_info["model_size"]}_{get_info["batchsize"]}_{get_info["lambda_memory"]}.json','w') as f:
        json.dump(get_info, f, ensure_ascii=False, indent=4)  
    s3_client.upload_file(f'/tmp/{get_info["model_name"]}_{get_info["model_size"]}_{get_info["batchsize"]}_{get_info["lambda_memory"]}.json',BUCKET_NAME,f'results/{get_info["optimizer"]}/{get_info["hardware"]}/{get_info["model_name"]}_{get_info["model_size"]}_{get_info["batchsize"]}_{get_info["lambda_memory"]}.json')

    return get_info

def ses_send(info):
    dst_format = {"ToAddresses":[f"{info['user_email']}"],
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
                                "Data": f"AYCI convert time results\n---------------------------------------\n{info['model_name']} convert using {info['optimizer'].upper()} on {info['hardware'].upper()} \n{info['model_name']} size : {info['model_size']} MB\nConvert {info['model_name']} latency : {round(info['convert_time'],4)} s\n\nAYCI inference time results\n---------------------------------------\n{info['model_name']} inference Done!\n{info['model_name']} size : {info['model_size']} MB\nInference batchsize : {info['batchsize']}\nInference {info['model_name']} latency on {info['hardware'].upper()}: {round(info['inference_time'],4)} s\n-----------------------------------------------\nLambda memory size : {info['lambda_memory']}\nMax Memory Used : {info['max_memory_used']}",
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
            info = {
                'model_name':event[i]['model_name'],
                'model_size':event[i]['model_size'],
                'hardware':event[i]['hardware'],
                'framework':event[i]['framework'],
                'optimizer':event[i]['optimizer'],
                'lambda_memory':event[i]['lambda_memory'],
                'batchsize':event[i]['batchsize'],
                'user_email':event[i]['user_email'],
                'request_id':event[i]['request_id'],
                'log_group_name':event[i]['log_group_name']
            }
            max_memory_used = getMemoryUsed(info)
            print(max_memory_used)

            get_info = upload_data(info,max_memory_used)
#             ses_send(get_info)

        else:
            pass


    return { 'result':'upload done'}
