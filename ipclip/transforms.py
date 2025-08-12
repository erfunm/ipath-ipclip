from torchvision.transforms import (
    RandomAffine, RandomPerspective, RandomHorizontalFlip,
    Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode,
)
BICUBIC = InterpolationMode.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def train_transform(first_resize: int = 512, n_px: int = 224):
    return Compose([
        Resize([first_resize], interpolation=InterpolationMode.BICUBIC),
        CenterCrop(first_resize),
        RandomHorizontalFlip(),
        RandomAffine(
            degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1),
            shear=(-10, 10, -10, 10), interpolation=InterpolationMode.BILINEAR, fill=127,
        ),
        RandomPerspective(distortion_scale=0.25, p=0.3, interpolation=InterpolationMode.BILINEAR, fill=127),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def eval_transform(n_px: int = 224):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
