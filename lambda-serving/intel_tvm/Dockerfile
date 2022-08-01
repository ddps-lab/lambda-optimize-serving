FROM amazon/aws-lambda-python:3.8

# install essential library
RUN yum install -y wget
RUN yum -y install cmake3 gcc gcc-c++ make && ln -s /usr/bin/cmake3 /usr/bin/cmake
RUN yum -y install python3-dev python3-setuptools libtinfo-dev zlib1g-dev build-essential libedit-dev llvm llvm-devel libxml2-dev git tar gcc gcc-c++
RUN yum -y install libgomp gcc-gfortran libgfortran blas lapack atlas-sse3-devel

# git 특정 폴더만 clone
RUN mkdir lambda-optimize-serving
WORKDIR lambda-optimize-serving
RUN git init
RUN git config core.sparseCheckout true
RUN git remote add -f origin https://github.com/ddps-lab/lambda-optimize-serving
RUN echo lambda-serving/intel_tvm/* >> .git/info/sparse-checkout
RUN git pull origin main
WORKDIR lambda-serving/intel_tvm

# tvm env 설정 
ENV TVM_HOME=/var/task/lambda-optimize-serving/lambda-serving/intel_tvm/tvm
ENV PATH=$PATH:$TVM_HOME/bin
ENV PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
ENV PATH=$TVM_HOME/python:$PATH

RUN mkdir tvm/build && cp config.cmake tvm/build && env CC=cc CXX=CC

RUN cd tvm/build && cmake .. && make -j3

RUN pip3 install -r requirements.txt

RUN cp lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
