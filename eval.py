import torch
import torchvision
from mask.dataset import MaskedFaceTestDataset
from mask.model import load_model
from mask.metrics import model_evaluate

WEIGHTS_PATH = "weights/pretrained_epoch_10.pth"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

val_dataset = MaskedFaceTestDataset(
    "MaskedFace/val", img_transforms=torchvision.transforms.ToTensor()
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=4
)
print("Validation dataset size:", len(val_dataset))

model = load_model(WEIGHTS_PATH, device=device)
model.to(device)
print("Model loaded from", WEIGHTS_PATH)

mape = model_evaluate(val_loader, model, device)
print(f"Validation MAPE: {mape:.4f}%")
