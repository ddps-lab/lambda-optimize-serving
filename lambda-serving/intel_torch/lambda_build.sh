export IMAGE_NAME="serving_intel_torch"

sudo chmod 666 /var/run/docker.sock
sudo service docker start

docker build -t $IMAGE_NAME . --no-cache

export ACCOUNT_ID=$(aws sts get-caller-identity --output text --query Account)

docker tag $IMAGE_NAME $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com

aws ecr delete-repository \
    --repository-name $IMAGE_NAME \
    --force

sleep 5

aws ecr create-repository \
    --repository-name $IMAGE_NAME \
    --image-scanning-configuration scanOnPush=true \
    --region us-west-2

docker push $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME

aws lambda delete-function \
    --function-name $IMAGE_NAME

sleep 5

aws lambda create-function --region us-west-2 --function-name $IMAGE_NAME \
            --package-type Image  \
            --code ImageUri=$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME:latest   \
            --role arn:aws:iam::$ACCOUNT_ID:role/jg-efs-role \
            --architectures x86_64 \
            --memory-size 2048 \
            --timeout 120
