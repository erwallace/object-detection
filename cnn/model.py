import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)


def get_model(num_classes: int = 4) -> torch.nn.Module:
    """Load a pre-trained model and replace the head with a new one.

    Args:
        num_classes (int): Number of classes in the dataset.

    Returns:
        torch.nn.Module: The modified model.
    """
    # Load an object detection model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model(
    weights_path: str, num_classes: int = 4, device="cuda"
) -> torch.nn.Module:
    """Load a pre-trained model from a specified path.

    Args:
        weights_path (str): Path to the model weights.
        num_classes (int): Number of classes in the dataset.
        device (str): Device to load the model on ("cuda" or "cpu").

    Returns:
        torch.nn.Module: The loaded model.
    """
    if device == "cpu":
        map_location = torch.device("cpu")
    else:
        map_location = None

    # Load the model
    model = get_model(num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=map_location))
    return model
