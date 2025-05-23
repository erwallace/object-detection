import torch
import torchvision
from mask.dataset import MaskedFaceTestDataset

# from cnn.transforms import img_transforms
from mask.model import get_model
from mask.metrics import model_evaluate

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

num_classes = (
    4  # 3 classes (with_mask, without_mask, mask_weared_incorrect) + background
)

model = get_model(num_classes)
model.to(device)

val_dataset = MaskedFaceTestDataset(
    "MaskedFace/val", img_transforms=torchvision.transforms.ToTensor()
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=4
)
print("Validation dataset size:", len(val_dataset))

train_dataset = MaskedFaceTestDataset(
    "MaskedFace/train", img_transforms=torchvision.transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=True, num_workers=4
)
print("Train dataset size:", len(train_dataset))

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 10

print("Training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in targets.items()}]

        for target in targets:
            if target["boxes"].dim() == 3:
                target["boxes"] = target["boxes"].squeeze(0)
            if target["labels"].dim() == 2:
                target["labels"] = target["labels"].squeeze(0)

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass and optimization
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    mape = model_evaluate(val_loader, model, device)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader):.4f}, Validation MAPE: {mape:.4f}%"
    )

# Save the model's weights (state_dict)
save_path = "weights/testing_10_epoch.pth"
torch.save(model.state_dict(), save_path)
