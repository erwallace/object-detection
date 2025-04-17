import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torchvision
from torch.utils.data import DataLoader

from mask.metrics import mean_absolute_percentage_error, count_
from mask.model import get_model
from mask.dataset import MaskedFaceTestDataset

# Assuming these imports are needed based on your current code
# Add any missing imports from your original script


class MaskDetectionDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, batch_size=1, num_workers=4):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Define transforms
        self.transforms = torchvision.transforms.ToTensor()

        # Load datasets
        self.train_dataset = MaskedFaceTestDataset(
            self.train_path, img_transforms=self.transforms
        )

        self.val_dataset = MaskedFaceTestDataset(
            self.val_path, img_transforms=self.transforms
        )

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class MaskDetectionModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.best_mape = float("inf")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        # Convert to device and prepare targets like in your original code
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        for target in targets:
            if target["boxes"].dim() == 3:
                target["boxes"] = target["boxes"].squeeze(0)
            if target["labels"].dim() == 2:
                target["labels"] = target["labels"].squeeze(0)

        # Get loss dict from model
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        # Log losses
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=True)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # Similar processing as training_step
        images = [img.to(self.device) for img in images]
        # targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        predictions = self.model(images)

        mape = self.mean_absolute_percentage_error(predictions, targets)

        self.log("val_mape", mape, on_epoch=True)
        return {"val_mape": mape}

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        # Add a learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mape",  # Track the validation MAPE
                "interval": "epoch",
            },
        }

    def mean_absolute_percentage_error(self, predictions, targets):
        """Calculate the Mean Absolute Percentage Error (MAPE) between predictions and targets."""
        pred_counts = count_(predictions)
        true_counts = count_(targets)

        return mean_absolute_percentage_error(pred_counts, true_counts)


# Main training function
def main():
    # Initialize your model (assuming you already have this defined somewhere)
    model = get_model(num_classes=4)

    # Initialize data module
    data_module = MaskDetectionDataModule(
        train_path="MaskedFace/train", val_path="MaskedFace/val"
    )

    # Initialize the Lightning model
    lightning_model = MaskDetectionModel(model)

    # Checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        dirpath="weights/",
        filename="best-checkpoint-{epoch:02d}-{val_mape:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_mape",
        mode="min",
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",  # Automatically select available accelerator
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
    )

    # Train the model
    trainer.fit(lightning_model, data_module)

    print(f"Best model path: {checkpoint_callback.best_model_path}")
    print(f"Best validation MAPE: {checkpoint_callback.best_model_score}")


if __name__ == "__main__":
    main()
