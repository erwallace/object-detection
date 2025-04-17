# TODO: data augmentation and oversampling
# TODO: pad tensors to allow batch size > 1
import torch
import torchvision
import torchvision.transforms as T


def find_largest_dimensions(dataloaders: list) -> tuple:
    max_width, max_height = 0, 0
    for dataloader in dataloaders:
        for images, _ in dataloader:
            for img in images:
                # Convert tensor to PIL image if necessary
                if isinstance(img, torch.Tensor):
                    img = T.ToPILImage()(img)
                width, height = img.size
                max_width = max(max_width, width)
                max_height = max(max_height, height)
    return max_width, max_height


def img_transforms(train: bool, max_width:int=600, max_height:int=600) -> torchvision.transforms.Compose:
    """Get the transformation pipeline for the dataset.

    Args:
        train (bool): If True, apply training transformations.

    Returns:
        torchvision.transforms.Compose: The transformation pipeline.
    """
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms += [
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Pad((0, 0, max_width, max_height))
    ]

    if train:
        transforms += [
            # during training, randomly flip the training images
            # and ground-truth for data augmentation
            T.RandomHorizontalFlip(1),
        ]

    # for debugging
    transforms.append(T.ToPILImage())
        
    return T.Compose(transforms)

# pad images to same size for batching


