export IMAGE_NAME="convert_torch_arm_tvm"

sudo service docker start
sudo chmod 666 /var/run/docker.sock

docker build -t $IMAGE_NAME . --no-cache

export ACCOUNT_ID=$(aws sts get-caller-identity --output text --query Account)

docker tag $IMAGE_NAME $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com

aws ecr create-repository \
    --repository-name $IMAGE_NAME \
    --image-scanning-configuration scanOnPush=true \
    --region us-west-2

docker push $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME