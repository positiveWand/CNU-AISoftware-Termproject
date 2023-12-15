from torchvision.transforms import Resize, InterpolationMode, RandomRotation, Compose, ToTensor, Normalize
from PIL import Image

def convert_img_to_rgb(image: Image):
    return image.convert("RGB")

def img_augment_transform(size: int, max_degrees: int):
    return Compose([
        Resize(size, interpolation=InterpolationMode.BICUBIC),
        RandomRotation(degrees=max_degrees),
        convert_img_to_rgb,
        ToTensor(),
        Normalize((0.6671, 0.6667, 0.6658), (0.2143, 0.2142, 0.2140))
    ])

def img_transform(size: int):
    return Compose([
        Resize(size, interpolation=InterpolationMode.BICUBIC),
        convert_img_to_rgb,
        ToTensor(),
        Normalize((0.6671, 0.6667, 0.6658), (0.2143, 0.2142, 0.2140))
    ])

def text_transform(before, after):
    return lambda text: text.replace(before, after)