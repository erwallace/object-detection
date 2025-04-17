from torch.utils.data import ConcatDataset, Dataset


class ListDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def calulate_class_distribution(dataset):
    with_mask = []
    without_mask = []
    mask_weared_incorrect = []

    for img, target in dataset:
        if 0 in target["labels"]:
            with_mask.append((img, target))
        if 1 in target["labels"]:
            without_mask.append((img, target))
        if 2 in target["labels"]:
            mask_weared_incorrect.append((img, target))
    print(
        f"with_mask: {len(with_mask)}, without_mask: {len(without_mask)}, mask_weared_incorrect: {len(mask_weared_incorrect)}"
    )
    print(
        f"with_mask: {100*len(with_mask)/len(dataset):.1f}%, without_mask: {100*len(without_mask)/len(dataset):.1f}%, mask_weared_incorrect: {100*len(mask_weared_incorrect)/len(dataset):.1f}%"
    )
    return with_mask, without_mask, mask_weared_incorrect


def oversample(dataset, iterations=1):
    print("Initial dataset")
    _, without_mask, mask_weared_incorrect = calulate_class_distribution(dataset)

    combined_dataset = ConcatDataset(
        [dataset, ListDataset(without_mask), ListDataset(mask_weared_incorrect)]
    )

    print("Oversampling dataset")
    calulate_class_distribution(combined_dataset)

    return combined_dataset
