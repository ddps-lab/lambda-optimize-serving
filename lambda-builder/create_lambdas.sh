#!/bin/bash

export IMAGE_NAME="serving"
export ACCOUNT_ID=$(aws sts get-caller-identity --output text --query Account)

lambda_memory="512 1024 2048 4096 8192 10240"
arm_serving="arm_onnx arm_torch arm_tvm"
intel_serving="intel_onnx intel_torch intel_tvm"

for serv in $arm_serving; do
  for lm in $lambda_memory; do
    aws lambda delete-function \
    --function-name $IMAGE_NAME'_'$serv'_'$lm

    sleep 5

    aws lambda create-function --region us-west-2 --function-name $IMAGE_NAME'_'$serv'_'$lm \
      --package-type Image \
      --code ImageUri=$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME'_'$serv:latest \
      --role arn:aws:iam::$ACCOUNT_ID:role/jg-efs-role \
      --architectures arm64 \
      --memory-size $lm \
      --timeout 240
  done
done

for serv in $intel_serving; do
  for lm in $lambda_memory; do
    aws lambda delete-function \
    --function-name $IMAGE_NAME'_'$serv'_'$lm

    sleep 5

    aws lambda create-function --region us-west-2 --function-name $IMAGE_NAME'_'$serv'_'$lm \
      --package-type Image \
      --code ImageUri=$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME'_'$serv:latest \
      --role arn:aws:iam::$ACCOUNT_ID:role/jg-efs-role \
      --architectures x86_64 \
      --memory-size $lm \
      --timeout 240
  done
done


