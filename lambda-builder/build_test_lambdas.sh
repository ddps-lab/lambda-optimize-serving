#!/bin/bash

export IMAGE_NAME="serving"
export ACCOUNT_ID=$(aws sts get-caller-identity --output text --query Account)

arm_serving="arm_onnx arm_torch arm_tvm"
intel_serving="intel_onnx intel_torch intel_tvm"

for serv in $arm_serving; do
  aws lambda delete-function \
    --function-name $IMAGE_NAME'_'$serv

  sleep 5

  aws lambda create-function --region us-west-2 --function-name $IMAGE_NAME'_'$serv \
    --package-type Image \
    --code ImageUri=$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME'_'$serv:latest \
    --role arn:aws:iam::$ACCOUNT_ID:role/jg-efs-role \
    --architectures arm64 \
    --memory-size 2048 \
    --timeout 240

  sleep 60

  aws lambda update-function-configuration --region us-west-2 --function-name $IMAGE_NAME'_'$serv \
    --environment "Variables={BUCKET_NAME=ayci}"
done

for serv in $intel_serving; do
  aws lambda delete-function \
    --function-name $IMAGE_NAME'_'$serv

  sleep 5

  aws lambda create-function --region us-west-2 --function-name $IMAGE_NAME'_'$serv \
    --package-type Image \
    --code ImageUri=$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME'_'$serv:latest \
    --role arn:aws:iam::$ACCOUNT_ID:role/jg-efs-role \
    --architectures x86_64 \
    --memory-size 2048 \
    --timeout 240

  sleep 60

  aws lambda update-function-configuration --region us-west-2 --function-name $IMAGE_NAME'_'$serv \
    --environment "Variables={BUCKET_NAME=ayci}"
done

export IMAGE_NAME="convert_torch"

aws lambda delete-function \
  --function-name $IMAGE_NAME'_intel_onnx'
aws lambda delete-function \
  --function-name $IMAGE_NAME'_arm_onnx'
aws lambda delete-function \
  --function-name $IMAGE_NAME'_intel_tvm'
aws lambda delete-function \
  --function-name $IMAGE_NAME'_arm_tvm'

sleep 5

aws lambda create-function --region us-west-2 --function-name $IMAGE_NAME'_intel_onnx' \
  --package-type Image \
  --code ImageUri=$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME'_intel_onnx':latest \
  --role arn:aws:iam::$ACCOUNT_ID:role/jg-efs-role \
  --architectures x86_64 \
  --memory-size 4096 \
  --timeout 240

aws lambda create-function --region us-west-2 --function-name $IMAGE_NAME'_arm_onnx' \
  --package-type Image \
  --code ImageUri=$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME'_arm_onnx':latest \
  --role arn:aws:iam::$ACCOUNT_ID:role/jg-efs-role \
  --architectures arm64 \
  --memory-size 4096 \
  --timeout 240

aws lambda create-function --region us-west-2 --function-name $IMAGE_NAME'_intel_tvm' \
  --package-type Image \
  --code ImageUri=$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME'_intel_tvm':latest \
  --role arn:aws:iam::$ACCOUNT_ID:role/jg-efs-role \
  --architectures x86_64 \
  --memory-size 4096 \
  --timeout 240

aws lambda create-function --region us-west-2 --function-name $IMAGE_NAME'_arm_tvm' \
  --package-type Image \
  --code ImageUri=$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME'_arm_tvm':latest \
  --role arn:aws:iam::$ACCOUNT_ID:role/jg-efs-role \
  --architectures arm64 \
  --memory-size 4096 \
  --timeout 240

sleep 60

converting="intel_onnx arm_onnx intel_tvm arm_tvm"

for serv in $converting; do
  aws lambda update-function-configuration --region us-west-2 --function-name $IMAGE_NAME'_'$serv \
    --environment "Variables={BUCKET_NAME=ayci}"
done
