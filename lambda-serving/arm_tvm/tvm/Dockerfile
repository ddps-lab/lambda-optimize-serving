FROM public.ecr.aws/lambda/python:3.8-arm64

# install essential library
RUN yum install -y wget
RUN yum -y install cmake3 gcc gcc-c++ make && ln -s /usr/bin/cmake3 /usr/bin/cmake
RUN yum -y install python3-dev python3-setuptools libtinfo-dev zlib1g-dev build-essential libedit-dev llvm llvm-devel libxml2-dev git tar wget gcc gcc-c++

# git clone
RUN git clone https://github.com/jaeriver/DL_Lambda_Serving
WORKDIR DL_Lambda_Serving/arm/tvm

# ENV PYTHONPATH "/var/lang/lib/python3.8/site-packages:/var/task"
ENV TVM_HOME=/var/task/DL_Lambda_Serving/arm/tvm/tvm
ENV PATH=$PATH:$TVM_HOME/bin
ENV PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
ENV PATH=$TVM_HOME/python:$PATH

RUN pip3 install -r requirements.txt

# install packages
RUN mkdir tvm/build && cp config.cmake tvm/build && env CC=cc CXX=CC

RUN cd tvm/build && cmake .. && make -j3

RUN cp lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]

