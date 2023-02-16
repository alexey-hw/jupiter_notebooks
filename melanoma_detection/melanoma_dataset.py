import os
from PIL import Image
from torch.utils.data import Dataset


def find_images(dir_path, validate=None):
    data_paths = []
    for root, _, image_names in os.walk(dir_path):
        for image_name in image_names:
            if validate is not None:
                if validate(image_name):
                    data_paths.append(os.path.join(root, image_name))
                else:
                    print(os.path.join(root, image_name) + " label is unrecognized, image skipped")
            else:
                data_paths.append(os.path.join(root, image_name))
    data_paths.sort()
    return data_paths


class MelanomaUnlabeledDataset(Dataset):

    def __init__(self, dir_path, transform=None):
        super().__init__()
        self.image_paths = find_images(dir_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        if self.transform is not None:
            return self.transform(Image.open(image_path))
        return Image.open(image_path)


class MelanomaLabeledDataset(Dataset):

    def __init__(self, dir_path, transform=None):
        super().__init__()
        self.image_paths = find_images(dir_path, lambda x: x.find("_0.") != -1 or x.find("_1.") != -1)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_label = 0 if image_path.rfind("_0.") != -1 else 1
        if self.transform is not None:
            return self.transform(Image.open(image_path)), image_label
        return Image.open(image_path), image_label
