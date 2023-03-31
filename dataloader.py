import torch
import torchvision
from torchvision import transforms 

def load_transformed_dataset(cfg):
    data_transforms = [
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.StanfordCars(
        root=".", download=True, transform=data_transform
        )

    test = torchvision.datasets.StanfordCars(root=".", download=True, 
                                            transform=data_transform, split='test')
    return torch.utils.data.ConcatDataset([train, test])