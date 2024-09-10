import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from PIL import Image
import numpy as np

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(p=0.2),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0)),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def augment_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    augmented = transform(image=image)
    return augmented["image"]

# Example usage
augmented_image = augment_image('/Users/nishantarora/Desktop/intern proj/Machine2.jpg')
