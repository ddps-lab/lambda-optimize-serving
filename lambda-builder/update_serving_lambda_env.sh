#!/bin/bash
bucket_name=$1

export IMAGE_NAME="serving"
export ACCOUNT_ID=$(aws sts get-caller-identity --output text --query Account)

lambda_memory="512 1024 2048 4096 8192 10240"
arm_serving="arm_onnx arm_torch arm_tvm"
intel_serving="intel_onnx intel_torch intel_tvm"

for serv in $arm_serving; do
  for lm in $lambda_memory; do
    aws lambda update-function-configuration --region us-west-2 --function-name $IMAGE_NAME'_'$serv'_'$lm \
      --environment "Variables={BUCKET_NAME=$bucket_name}"
  done
done

for serv in $intel_serving; do
  for lm in $lambda_memory; do
    aws lambda update-function-configuration --region us-west-2 --function-name $IMAGE_NAME'_'$serv'_'$lm \
      --environment "Variables={BUCKET_NAME=$bucket_name}"
  done
done
