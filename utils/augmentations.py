from torchvision import transforms

def get_train_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
    ])

def get_val_transforms():
    from torchvision import transforms as T
    return T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

