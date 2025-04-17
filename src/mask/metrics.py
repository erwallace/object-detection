import torch


def model_evaluate(test_loader, model, device="cuda") -> float:
    """Evaluate the model on the test dataset and calculate MAPE."""
    model.eval()

    with torch.no_grad():
        predictions = []
        truths = []
        for images, targets in test_loader:
            images = [img.to(device) for img in images]

            prediction = model(images)
            predictions.append(prediction)
            truths.append(targets)

        pred_counts = count_(predictions)
        true_counts = count_(truths)

        return mean_absolute_percentage_error(pred_counts, true_counts)


def count_(targets) -> torch.Tensor:
    """Count the number of masks in the test dataset."""
    if isinstance(targets[0], list) and len(targets[0]) == 1:
        targets = [t[0] for t in targets]

    return torch.stack(
        [
            torch.tensor(
                [
                    label.tolist().count(0),
                    label.tolist().count(1),
                    label.tolist().count(2),
                ]
            )
            for label in (t["labels"] for t in targets)
        ]
    )


def mean_absolute_percentage_error(
    y_true: torch.Tensor, y_pred: torch.Tensor, epsilon=1e-10
) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between two Nx3 tensors.

    Args:
        y_true (torch.Tensor): Ground truth tensor of shape (N, 3).
        y_pred (torch.Tensor): Predicted tensor of shape (N, 3).
        epsilon (float): Small value to avoid division by zero.

    Returns:
        float: The MAPE value.
    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    assert y_true.shape[1] == 3, "y_true and y_pred must have 3 columns for 3 classes"

    # Calculate absolute percentage error
    ape = torch.abs(y_true - y_pred) / (y_true + epsilon)

    # Compute the mean of all absolute percentage errors
    mape = torch.mean(ape) * 100  # Multiply by 100 to express as a percentage
    return mape.item()  # Convert the result to a Python float
