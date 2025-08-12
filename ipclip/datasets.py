from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class CLIPImageCaptioningDataset(Dataset):
    def __init__(self, df, preprocessing):
        self.images = df["image"].tolist()
        self.caption = df["caption"].tolist()
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        img = self.preprocessing(Image.open(self.images[idx]).convert('RGB'))
        cap = self.caption[idx]
        return img, cap


class CLIPCaptioningDataset(Dataset):
    def __init__(self, captions):
        self.captions = captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx]


class CLIPImageDataset(Dataset):
    def __init__(self, image_paths, preprocessing):
        self.images = image_paths
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.preprocessing(Image.open(self.images[idx]).convert('RGB'))


class CLIPImageLabelDataset(Dataset):
    def __init__(self, df, preprocessing):
        self.images = df["image"].tolist()
        self.labels = df["label"].tolist()
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.preprocessing(Image.open(self.images[idx]).convert('RGB'))
        return img, self.labels[idx]
