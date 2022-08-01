#!/bin/bash

export IMAGE_NAME="serving"
export ACCOUNT_ID=$(aws sts get-caller-identity --output text --query Account)

sudo chmod 666 /var/run/docker.sock
sudo service docker start

serving="arm_onnx arm_torch arm_tvm"

for serv in $serving; do
  docker build -f "../lambda-serving/$serv/Dockerfile" -t $IMAGE_NAME'_'$serv . --no-cache
  docker tag $IMAGE_NAME'_'$serv $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME'_'$serv
  aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
  aws ecr delete-repository \
    --repository-name $IMAGE_NAME'_'$serv \
    --force

  sleep 5

  aws ecr create-repository \
    --repository-name $IMAGE_NAME'_'$serv \
    --image-scanning-configuration scanOnPush=true \
    --region us-west-2

  sleep 5

  docker push $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME'_'$serv
done

export IMAGE_NAME="convert_torch"

converting="arm_tvm"

for serv in $converting; do
  docker build -f "../lambda-optimize/$serv/Dockerfile" -t $IMAGE_NAME'_'$serv . --no-cache
  docker tag $IMAGE_NAME'_'$serv $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME'_'$serv
  aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
  aws ecr delete-repository \
  --repository-name $IMAGE_NAME'_'$serv \
  --force

  sleep 5

  aws ecr create-repository \
    --repository-name $IMAGE_NAME'_'$serv \
    --image-scanning-configuration scanOnPush=true \
    --region us-west-2
  docker push $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME'_'$serv
done