from cnn.metrics import _count_masks, count_, mean_absolute_percentage_error
from cnn.model import load_model
import torch
from torch.utils.data import DataLoader

def count_masks(test_dataset, model_path: str, device: str = "cuda") -> torch.Tensor:
    """Count the number of masks in the test dataset using a pre-trained model.
    
    Args:
        test_dataset (Dataset): The test dataset.
        model_path (str): Path to the pre-trained model.
    
    Returns:
        torch.Tensor: Nx3 array of count for each class.
    """
    model = load_model(model_path, device=device)
    model.to(device)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()

    with torch.no_grad():
        predictions = []
        for images, _ in test_loader:
            images = [img.to(device) for img in images]

            prediction = model(images)
            predictions.append(prediction)

    return count_(predictions)


if __name__ == "__main__":
    from cnn.dataset import MaskedFaceDataset
    import torchvision

    # Load the test dataset
    val_dataset = MaskedFaceDataset('MaskedFace/val', img_transforms=torchvision.transforms.ToTensor())
    WEIGHTS_PATH = 'models/pretrained_epoch_10.pth'
    
    # Count masks in the test dataset
    counts = count_masks(val_dataset, WEIGHTS_PATH)
    
    print(f"Counts of masks in the test dataset ({counts.shape}):")
    print(counts)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        truths = []
        for _, targets in val_loader:
            truths.append(targets)
        true_counts = count_(truths)

    mape = mean_absolute_percentage_error(counts, true_counts)
    print(f"Mean Absolute Percentage Error: {mape:.4f}%")

    

