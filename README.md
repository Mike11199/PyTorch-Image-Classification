## PyTorch Image Classification Project

- Front End Repo
  - https://github.com/Mike11199/PyTorch-Image-Classification-TypeScript 

- Personal project.  Deployed a SageMaker endpoint and API Gateway using a pre-trained PyTorch fasterrcnn_resnet50_fpn_v2 computer vision model.

## Deployed AWS SageMaker Endpoint

![image](https://github.com/Mike11199/PyTorch-Image-Classification/assets/91037796/5e9fdef7-d082-45bd-b672-3dc7e8e4d415)

## Lambda to Invoke Endpoint

- https://docs.aws.amazon.com/apigateway/latest/developerguide/getting-started-with-lambda-integration.html 

![image](https://github.com/Mike11199/PyTorch-Image-Classification/assets/91037796/ad98ae5f-abc3-4279-9744-1ef7604af448)

## AWS API Gateway to Invoke Lambda from Outside AWS VPC

- This was difficult.  The image had to be sent in binary form to the API Gateway in the request blob and handled in the lambda as base64.
- Before, my lambda took an image url, downloaded it with urllib and converted to base 64.  Now it simple takes the binary input of the image.

![image](https://github.com/Mike11199/PyTorch-Image-Classification/assets/91037796/0dd676cc-fb66-4308-a159-40f0029bf2c7)

![image](https://github.com/Mike11199/PyTorch-Image-Classification/assets/91037796/12e436af-8abe-4fad-9a47-fec6a5b486ab)

![image](https://github.com/Mike11199/PyTorch-Image-Classification/assets/91037796/31e15c2d-09d8-496c-bb34-ad2e906c77a1)

![image](https://github.com/Mike11199/PyTorch-Image-Classification/assets/91037796/8e5f60ef-80de-4925-b65c-f3dd32fabefb)


## Custom inference.py Handler

- https://sagemaker.readthedocs.io/en/v2.29.1/frameworks/pytorch/using_pytorch.html
- Reference a bit of https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_applying_machine_learning/sagemaker_pytorch_model_zoo/sagemaker_pytorch_model_zoo.ipynb
- Large parts of the inference.py had to be rewritten due to issues with cv2 and tensors.  About ~12 hours to get the endpoint up and running.

![image](https://github.com/Mike11199/PyTorch-Image-Classification/assets/91037796/52ac1269-ccb2-4964-9514-7b6208b78d03)

## Notebook Screenshots

![image](https://github.com/Mike11199/PyTorch-Image-Classification/assets/91037796/00f269c1-d941-4cdc-b6a0-948e29ad63f4)

![image](https://github.com/Mike11199/PyTorch-Image-Classification/assets/91037796/d9369c98-e610-4d0b-903a-cde9d1dde618)









