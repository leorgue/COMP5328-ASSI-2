# Custom Dataset class
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CIFARDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images) / 255.0
        self.images = self.images.permute(0, 3, 1, 2)
        self.labels = torch.LongTensor(labels)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.normalize(self.images[idx])
        return image, self.labels[idx], idx


# Custom Dataset class for FashionMNIST
class FashionMNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images) / 255.0
        # Handle grayscale images - add channel dimension if needed
        if len(self.images.shape) == 3:  # (N, H, W)
            self.images = self.images.unsqueeze(1)  # (N, 1, H, W)
        elif len(self.images.shape) == 4 and self.images.shape[-1] == 1:  # (N, H, W, 1)
            self.images = self.images.permute(0, 3, 1, 2)  # (N, 1, H, W)
        self.labels = torch.LongTensor(labels)
        # FashionMNIST normalization (mean and std for grayscale)
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.normalize(self.images[idx])
        return image, self.labels[idx], idx