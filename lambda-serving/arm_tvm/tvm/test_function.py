import time
import_start_time = time.time()
from tvm import relay
import numpy as np 
import tvm 
from tvm.contrib import graph_executor
from tvm.contrib import graph_runtime
import onnx
import boto3
import json
print('import time: ', time.time() - import_start_time)

image_size = 224
channel = 3
image_classification_shape_type = {
    "mxnet" : (channel, image_size, image_size),
    "tf" : (image_size, image_size, channel)
}

s3_client = boto3.client('s3')    

def get_model(bucket_name, model_path):
    model_params = s3_client.get_object(Bucket=bucket_name, Key=model_path + '/model.params')['Body'].read()
    model_json = s3_client.get_object(Bucket=bucket_name, Key=model_path + '/model.json')['Body'].read().decode('utf-8')
    model_lib = s3_client.get_object(Bucket=bucket_name, Key=model_path + '/model.tar')['Body'].read()
    return model_params, model_json, model_lib

def make_dataset(batch_size, workload, framework):
    if workload == "image_classification":
        image_shape = image_classification_shape_type[framework]
        data_shape = (batch_size,) + image_shape

        data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

        return data, image_shape
    # case bert
    else:
        seq_length = 128
        shape_dict = {
            "data0": (batch_size, seq_length),
            "data1": (batch_size, seq_length),
            "data2": (batch_size,),
        }
        dtype = "float32"
        inputs = np.random.randint(0, 2000, size=(batch_size, seq_length)).astype(dtype)
        token_types = np.random.uniform(size=(batch_size, seq_length)).astype(dtype)
        valid_length = np.asarray([seq_length] * batch_size).astype(dtype)

        return inputs, token_types, valid_length

load_model = time.time()


def lambda_handler(event, context):
    bucket_name = event['bucket_name']
    batch_size = event['batch_size']
    arch_type = event['arch_type']
    framework = event['framework']
    model_name = event['model_name']
    compiler = 'tvm'
    model_path = f'{framework}/{compiler}/{arch_type}/{model_name}'
    workload = event['workload']
    is_build = event['is_build']
    count = event['count']
    
    if arch_type == 'arm':
        target = tvm.target.arm_cpu()
    else:
        target = arch_type
    ctx = tvm.cpu()
    
    params, graph, lib = get_model(bucket_name, model_path)
    print(type(params), type(graph), type(lib))
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input(**params)
    
    if workload == "image_classification":
        data, image_shape = make_dataset(batch_size, workload, framework)
        input_name = "data"
        module.set_input(input_name, data)
    #case bert
    else:
        data, token_types, valid_length = make_dataset(batch_size, workload, framework)
        module.set_input(data0=data, data1=token_types, data2=valid_length)
    
    
    time_list = []
    start_time = time.time()
    module.run(data=data)
    running_time = time.time() - start_time
    print(f"TVM {model_name}-{batch_size} inference latency : ",(running_time)*1000,"ms")
    return running_time

  
event = {
  "bucket_name": "dl-converted-models",
  "batch_size": 1,
  "arch_type": "intel",
  "framework" : "mxnet",
  "model_name": "bert_base_1",
  "workload" : "bert",
  "count": 5,
  "is_build":"false"
}
print(lambda_handler(event,""))
