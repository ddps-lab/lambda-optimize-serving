FROM public.ecr.aws/lambda/python:3.8-arm64

# install essential library
RUN yum install -y wget
RUN yum -y install cmake3 gcc gcc-c++ make && ln -s /usr/bin/cmake3 /usr/bin/cmake
RUN yum -y install python3-dev python3-setuptools libtinfo-dev zlib1g-dev build-essential libedit-dev llvm llvm-devel libxml2-dev git tar gcc gcc-c++
RUN yum -y install libgomp gcc-gfortran libgfortran blas lapack atlas-sse3-devel
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN mkdir -p /var/task/lib
RUN cp /usr/lib64/atlas-sse3/liblapack.so.3 /var/task/lib/. & cp /usr/lib64/atlas-sse3/libptf77blas.so.3 /var/task/lib/. & cp /usr/lib64/atlas-sse3/libf77blas.so.3 /var/task/lib/. & cp /usr/lib64/atlas-sse3/libptcblas.so.3 /var/task/lib/. & cp /usr/lib64/atlas-sse3/libcblas.so.3 /var/task/lib/. & cp /usr/lib64/atlas-sse3/libatlas.so.3 /var/task/lib/. & cp /usr/lib64/atlas-sse3/libptf77blas.so.3 /var/task/lib/. & cp /usr/lib64/libgfortran.so.3 /var/task/lib/. & cp /usr/lib64/libquadmath.so.0 /var/task/lib/.

# git 특정 폴더만 clone
RUN mkdir lambda-optimize-serving
WORKDIR lambda-optimize-serving
RUN git init
RUN git config core.sparseCheckout true
RUN git remote add -f origin https://github.com/ddps-lab/lambda-optimize-serving
RUN echo lambda-serving/arm/* >> .git/info/sparse-checkout
RUN git pull origin main
WORKDIR lambda-serving/intel

# tvm env 설정 
ENV TVM_HOME=/var/task/lambda-optimize-serving/lambda-serving/arm/tvm
ENV PATH=$PATH:$TVM_HOME/bin
ENV PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
ENV PATH=$TVM_HOME/python:$PATH

RUN mkdir tvm/build && cp config.cmake tvm/build && env CC=cc CXX=CC

RUN cd tvm/build && cmake .. && make -j3

RUN pip3 install -r requirements.txt

RUN cp lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
