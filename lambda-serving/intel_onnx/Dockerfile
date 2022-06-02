FROM amazon/aws-lambda-python:3.8

# install essential library
RUN yum -y install wget python3-dev python3-setuptools libtinfo-dev zlib1g-dev build-essential libedit-dev libxml2-dev git tar gcc gcc-c++

# git 특정 폴더만 clone
RUN mkdir lambda-optimize-serving
WORKDIR lambda-optimize-serving
RUN git init
RUN git config core.sparseCheckout true
RUN git remote add -f origin https://github.com/ddps-lab/lambda-optimize-serving
RUN echo lambda-serving/intel/* >> .git/info/sparse-checkout
RUN git pull origin main
WORKDIR lambda-serving/intel_onnx

RUN pip3 install -r requirements.txt

RUN cp lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
