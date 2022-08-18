## Build images for convert and serving lambda 

### How to do  
```
# 1. build convert & serving lambda images 
bash build_arm_images.sh True True 
bash build_intel_images.sh True True 

#1-1. build only covert lambda images 
bash build_arm_images.sh False True
bash build_intel_images.sh False True 

#1-2. build only serving lambda images 
bash build_arm_images.sh True False
bash build_intel_images.sh True False

#2. publish serving lambda with build images 
bash build_publish_lambdas.sh 

#3. update serving lambda environments 
bash update_serving_lambda_env.sh

```
