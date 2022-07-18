# inference results check module
import json
import os
import boto3
from datetime import datetime, timedelta
import time
import io
import pickle


BUCKET_NAME = os.environ.get('BUCKET_NAME')
s3_client = boto3.client('s3') 
s3 = boto3.resource('s3') 


def check_results(info,optimizer,hardware):    
    exist=False

    model_size = info['model_size']
    model_name = info['model_name']
    batchsize = info['batchsize']
    lambda_mem = info['lambda_memory']


    prefix = f'results/{optimizer}/{hardware}/'
    obj_list = s3_client.list_objects(Bucket=BUCKET_NAME,Prefix=prefix)

    check = prefix + f'{model_name}_{model_size}_{batchsize}_{lambda_mem}.json'
    contents_list = obj_list['Contents']
    for content in contents_list:
        # print(content)
        if content['Key']== check : 
            exist=True
            # 파일 내용을 읽어서 ses 보내기 
            obj = s3.Object(BUCKET_NAME,f"{check}")
            bytes_value = obj.get()['Body'].read()
            filejson = bytes_value.decode('utf8')
            fileobj = json.loads(filejson)
            ses_send(info['user_email'],fileobj,optimizer,hardware)

    return exist 

def ses_send(user_email,info , optimizer,hardware):
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
    body = json.loads(event['body'])
    print(body)
    info = {
            'model_name':body['model_name'],
            'model_size':body['model_size'],
            'hardware':body['hardware'],
            'framework':body['framework'],
            'optimizer':body['optimizer'],
            'lambda_memory':body['lambda_memory'],
            'batchsize':body['batchsize'],
            'user_email':body['user_email']
        }

    exist = check_results(info,info['optimizer'],info['hardware'])
    

    return { 'result_exist':exist}


