## PyTorch Image Classification Project

- Front End Repo
  - https://github.com/Mike11199/PyTorch-Image-Classification-TypeScript 

- Personal project.  Deployed a SageMaker endpoint and API Gateway using a pre-trained PyTorch fasterrcnn_resnet50_fpn_v2 computer vision model.

## Deployed on A SageMaker Endpoint

![image](https://github.com/Mike11199/PyTorch-Image-Classification/assets/91037796/5e9fdef7-d082-45bd-b672-3dc7e8e4d415)

## Lambda to Invoke Endpoint

- https://docs.aws.amazon.com/apigateway/latest/developerguide/getting-started-with-lambda-integration.html 

![image](https://github.com/Mike11199/PyTorch-Image-Classification/assets/91037796/ad98ae5f-abc3-4279-9744-1ef7604af448)

## Custom inference.py Handler

- https://sagemaker.readthedocs.io/en/v2.29.1/frameworks/pytorch/using_pytorch.html
- Reference a bit of https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_applying_machine_learning/sagemaker_pytorch_model_zoo/sagemaker_pytorch_model_zoo.ipynb
- Large parts of the inference.py had to be rewritten due to issues with cv2 and tensors.  About ~12 hours to get the endpoint up and running.

![image](https://github.com/Mike11199/PyTorch-Image-Classification/assets/91037796/52ac1269-ccb2-4964-9514-7b6208b78d03)

## Notebook Screenshots

![image](https://github.com/Mike11199/PyTorch-Image-Classification/assets/91037796/00f269c1-d941-4cdc-b6a0-948e29ad63f4)

![image](https://github.com/Mike11199/PyTorch-Image-Classification/assets/91037796/d9369c98-e610-4d0b-903a-cde9d1dde618)





## API Gateway to Invoke Lambda from Outside AWS VPC

![image](https://github.com/Mike11199/PyTorch-Image-Classification/assets/91037796/e7f81d05-7cb5-4b5d-9f85-e12f74bc1ecb)

- Postman Test of URL to call SageMaker endpoint

  ![image](https://github.com/Mike11199/PyTorch-Image-Classification/assets/91037796/1a9f4619-7510-4c7d-be93-cce894dc63b1)

![image](https://github.com/Mike11199/PyTorch-Image-Classification/assets/91037796/a66fa906-04f4-45db-8bd2-74c55ecceb6e)






