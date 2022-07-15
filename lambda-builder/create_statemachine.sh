lambda_memory="512 1024 2048 4096 8192 10240"

for lm in $lambda_memory; do
  aws stepfunctions create-state-machine --name "ayci_"$lm \
      --region us-wset-2 \
      --role-arn arn:aws:iam::741926482963:role/service-role/StepFunctions-Serverless_Inference_System-role-a3b1f8ca \
      --definition "$(cat "statemachine-"$lm".json")"

  done