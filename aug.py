import cv2
import torch
import albumentations as albu


def train_augmentation():
    train_transform = [
        albu.Resize(256, 256, interpolation=cv2.INTER_AREA, p=1),
        albu.HorizontalFlip(p=0.5),
        albu.RandomBrightness(p=0.5),
    ]
    return albu.Compose(train_transform)


def val_augmentation():
    train_transform = [
        albu.Resize(256, 256, interpolation=cv2.INTER_AREA, p=1),
    ]
    return albu.Compose(train_transform)


def mixup(data, targets, alpha=1, n_classes=18):
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets