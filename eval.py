import torch
import torchvision
from cnn.dataset import MaskedFaceDataset
from cnn.model import load_model
from cnn.metrics import model_evaluate

WEIGHTS_PATH = 'models/pretrained_epoch_10.pth'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

val_dataset = MaskedFaceDataset('MaskedFace/val', img_transforms=torchvision.transforms.ToTensor())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
print("Validation dataset size:", len(val_dataset))

model = load_model(WEIGHTS_PATH)
model.to(device)
print("Model loaded from", WEIGHTS_PATH)

mape = model_evaluate(val_loader, model, device)
print(f"Validation MAPE: {mape:.4f}%")