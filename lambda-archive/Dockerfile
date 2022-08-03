FROM amazon/aws-lambda-python:3.8

RUN /var/lang/bin/python3.8 -m pip install --upgrade pip
RUN yum -y update

# install essential library
RUN yum -y install  git 

# git 특정 폴더만 clone
RUN mkdir lambda-optimize-serving
WORKDIR lambda-optimize-serving
RUN git init
RUN git config core.sparseCheckout true
RUN git remote add -f origin https://github.com/ddps-lab/lambda-optimize-serving
RUN echo lambda-archive/* >> .git/info/sparse-checkout
RUN git pull origin main
WORKDIR lambda-archive

RUN pip3 install -r requirements.txt

RUN cp lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
