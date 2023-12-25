import os
import torch
import io
import logging
import numpy as np
import cv2
import base64
from io import BytesIO
import json
from PIL import Image
from torchvision import transforms

INFERENCE_ACCELERATOR_PRESENT_ENV = "SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT"
DEFAULT_MODEL_FILENAME = "model.pth"

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class ModelLoadError(Exception):
    pass


def _is_model_file(filename):
    is_model_file = False
    if os.path.isfile(filename):
        _, ext = os.path.splitext(filename)
        is_model_file = ext in [".pt", ".pth"]
    return is_model_file


def model_fn(model_dir):
    """Loads a model. For PyTorch, a default function to load a model only if Elastic Inference is used.
    In other cases, users should provide customized model_fn() in script.
    Args:
        model_dir: a directory where model is saved.
    Returns: A PyTorch model.
    """
    if os.getenv(INFERENCE_ACCELERATOR_PRESENT_ENV) == "true":
        model_path = os.path.join(model_dir, DEFAULT_MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Failed to load model with default model_fn: missing file {}.".format(
                    DEFAULT_MODEL_FILENAME
                )
            )
        # Client-framework is CPU only. But model will run in Elastic Inference server with CUDA.
        try:
            return torch.jit.load(model_path, map_location=torch.device("cpu"))
        except RuntimeError as e:
            raise ModelLoadError(
                "Failed to load {}. Please ensure model is saved using torchscript.".format(
                    model_path
                )
            ) from e
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join(model_dir, DEFAULT_MODEL_FILENAME)
        if not os.path.exists(model_path):
            model_files = [
                file for file in os.listdir(model_dir) if _is_model_file(file)
            ]
            if len(model_files) != 1:
                raise ValueError(
                    "Exactly one .pth or .pt file is required for PyTorch models: {}".format(
                        model_files
                    )
                )
            model_path = os.path.join(model_dir, model_files[0])
        try:
            model = torch.jit.load(model_path, map_location=device)
        except RuntimeError as e:
            raise ModelLoadError(
                "Failed to load {}. Please ensure model is saved using torchscript.".format(
                    model_path
                )
            ) from e
        model = model.to(device)
        logger.info('michael - loaded model!!!')
        return model


def input_fn(input_data, content_type):
    """
    Args:
        input_data: the request payload serialized in the content_type format
        content_type: the request content_type
    """
    logger.info("input_fn_start")
    decoded_data = base64.b64decode(input_data)
    # Convert the image data to a NumPy array
    image_array = np.frombuffer(decoded_data, np.uint8)

    # Decode the image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image_pil = Image.fromarray(image)
    print("Image shape before normalization:", image_pil.size)

    preprocess = transforms.Compose([transforms.ToTensor()])
    normalized = preprocess(image_pil)
    normalized_np = normalized.numpy()
    #image_array_swapped = normalized_np.transpose((1, 2, 0))
    print("Normalized shape:", normalized_np.shape)

    logger.info("input_fn_end")
    return normalized_np


def predict_fn(data, model):
    """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
    Runs prediction on GPU if cuda is available.
    Args:
        data: input data (torch.Tensor) for prediction deserialized by input_fn
        model: PyTorch model loaded in memory by model_fn
    Returns: a prediction
    """
    logger.info("predict_fn_start")
    with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.eval().to(device)
            targets = None
            data = torch.as_tensor(data, device=device)
            output = model([data], targets)
 
    logger.info("predict_fn_end")
    return output


def output_fn(prediction, accept):
    """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.
    Args:
        prediction: a prediction result from predict_fn
        accept: type which the output data needs to be serialized
    Returns: output data serialized
    """
    logger.info("output_fn_start")
    
   
    output_data =prediction[1][0]
    boxes = output_data.get("boxes", [])
    pred_scores = np.array(output_data.get("scores", []))
    pred_bboxes = np.array(output_data.get("boxes", []))

    detection_threshold = 0.9
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    labels = np.array(output_data.get("labels", [])[:len(boxes)])
    pred_classes = [coco_names[i] for i in labels]

    logger.info(pred_scores)
    logger.info(boxes)
    logger.info(labels)
    logger.info(pred_classes)

    logger.info("Constructing JSON")

    response_data = {
        "scores": pred_scores.tolist(),
        "boxes": boxes.tolist(),
        "labels": labels.tolist(),
        "classes": pred_classes
    }

    response_json = json.dumps(response_data)
    logger.info("output_fn_end")

    return response_json




