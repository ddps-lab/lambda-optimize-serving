import time
import json
from json import load
import numpy as np
import os
import boto3

BUCKET_NAME = os.environ.get('BUCKET_NAME')


def load_model(model_name, model_size):
    load_start = time.time()
    s3_client = boto3.client('s3')

    os.makedirs(os.path.dirname(f'/tmp/tvm/'), exist_ok=True)
    s3_client.download_file(BUCKET_NAME, f'models/tvm/arm/{model_name}_{model_size}.tar',
                            f'/tmp/tvm/{model_name}_{model_size}.tar')

    model = f"/tmp/tvm/{model_name}_{model_size}.tar"
    print(time.time() - load_start)
    return model


def tvm_serving(model_name, model_size, batchsize, imgsize=224, repeat=10):
    tvm_time = time.time()
    import tvm
    from tvm import relay
    import tvm.contrib.graph_executor as runtime
    print("tvm import time : ",time.time()-tvm_time)

    input_name = "input0"
    if model_name == "inception_v3":
        imgsize == 299
    input_shape = (batchsize, 3, imgsize, imgsize)
    output_shape = (batchsize, 1000)
    
    load_time = time.time()
    model_path = load_model(model_name, model_size)
    loaded_lib = tvm.runtime.load_module(model_path)
    print("model_load_time : ",time.time()-load_time)
    
    #target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"
    #dev = tvm.device(target, 0)
    dev = tvm.cpu()
    module = runtime.GraphModule(loaded_lib["default"](dev))
    data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
    data = tvm.nd.array(data, dev)
    module.set_input(input_name, data)
    
    time_list = []
    for i in range(repeat):
        start_time = time.time()
        module.run(data=data)
        running_time = time.time() - start_time
        time_list.append(running_time)
    print("tvm_end2end_time : ",time.time()-tvm_time)
    
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
                                "Data": f"AYCI convert time results\n---------------------------------------\n{info['model_name']} convert using TVM on ARM \n{info['model_name']} size : {info['model_size']} MB\nConvert {info['model_name']} latency : {round(info['convert_time'],4)} s\n\nAYCI inference time results\n---------------------------------------\n{info['model_name']} inference Done!\n{info['model_name']} size : {info['model_size']} MB\nLambda memory size : {info['lambda_memory']}\nInference batchsize : {info['batchsize']}\nInference {info['model_name']} latency on ARM: {round(info['inference_time'],4)} s",
                                "Charset": "UTF-8"
                            },
                        }
                    }
    mfile_path = "/tmp/message.json"

    with open(mfile_path, 'w', encoding='utf-8') as mfile:
        json.dump(message_format, mfile)

    os.system("aws ses send-email --from allyoucaninference@gmail.com --destination=file:///tmp/destination.json --message=file:///tmp/message.json")


def lambda_handler(event, context):
    start_time = time.time()
    model_name = event['model_name']
    model_size = event['model_size']
    hardware = event['hardware']
    framework = event['framework']
    optimizer = event['optimizer']
    lambda_memory = event['lambda_memory']
    batchsize = event['batchsize']
    user_email = event['user_email']
    convert_time = event['convert_time']
    request_id = context.aws_request_id
    log_group_name = context.log_group_name

    info = {
            'model_name': model_name,
            'model_size': model_size,
            'hardware': "intel",
            'framework': framework,
            'optimizer': "tvm",
            'lambda_memory': lambda_memory,
            'batchsize': batchsize,
            'user_email': user_email,
            'execute': True,
            'convert_time': convert_time,
            'inference_time': 0
        }

    if "tvm" in optimizer and "arm" in hardware:
        start_time = time.time()
        res = tvm_serving(model_name, model_size, batchsize)
        print('res: ', res)
        running_time = time.time() - start_time
        info['inference_time'] = running_time

        ses_send(user_email,info)

        return {
            'model_name': model_name,
            'model_size': model_size,
            'hardware': "arm",
            'framework': framework,
            'optimizer': "tvm",
            'lambda_memory': lambda_memory,
            'batchsize': batchsize,
            'user_email': user_email,
            'execute': True,
            'convert_time': convert_time,
            'request_id' : request_id,
            'log_group_name' : log_group_name
            'inference_time': res,
            'handler_time':running_time
        }
    else:
        return {
            'execute': False
        }
