import os
import cv2
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms import Resize


class CustomDataset(Dataset):
    def __init__(self, image_path, mask_path, pick_path, place_path, csv_path, dataset_type):
        self.samples = []
        self.image_path = image_path
        self.mask_path = mask_path
        self.pick_path = pick_path
        self.place_path = place_path
        self.csv_path = csv_path
        self.dataset_type = dataset_type
        self.image_files = os.listdir(image_path)
        self.target_size = (240, 320)
        self.read_data()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, mask, pick_dot, place_dot, label = self.samples[idx]

        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = image_transform(image)

        mask_transform = transforms.Compose([transforms.ToTensor()])
        mask = mask_transform(mask)

        pick_dot_transform = transforms.Compose([transforms.ToTensor()])
        pick_dot = pick_dot_transform(pick_dot)

        place_dot_transform = transforms.Compose([transforms.ToTensor()])
        place_dot = place_dot_transform(place_dot)

        return (image, mask, pick_dot, place_dot, label)

    def read_datapoint(self, datapoint):
        map_int = lambda x: np.array(list(map(int, x)))

        img_name = os.path.join(self.image_path, datapoint['image_name'])
        image = cv2.imread(img_name)
        image = Image.fromarray(image)
        image_resize_transform = Resize(self.target_size)
        image = image_resize_transform(image)

        mask_name = os.path.join(self.mask_path, datapoint['mask_name'])
        mask = cv2.imread(mask_name)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = Image.fromarray(mask)
        mask_resize_transform = Resize(self.target_size)
        mask = mask_resize_transform(mask)

        pick_name = os.path.join(self.pick_path, datapoint['pick_name'])
        pick_dot = cv2.imread(pick_name)
        pick_dot = cv2.cvtColor(pick_dot, cv2.COLOR_BGR2GRAY)
        pick_dot = Image.fromarray(pick_dot)
        pick_dot_resize_transform = Resize(self.target_size)
        pick_dot = pick_dot_resize_transform(pick_dot)

        place_name = os.path.join(self.place_path, datapoint['place_name'])
        place_dot = cv2.imread(place_name)
        place_dot = cv2.cvtColor(place_dot, cv2.COLOR_BGR2GRAY)
        place_dot = Image.fromarray(place_dot)
        place_dot_resize_transform = Resize(self.target_size)
        place_dot = place_dot_resize_transform(place_dot)

        label = datapoint['labels'][1:-1]
        label = map_int(label)
        label = torch.as_tensor(label, dtype=torch.float32)

        return (image, mask, pick_dot, place_dot, label)

    def read_data(self):
        with open(self.csv_path, newline='') as data_csv:
            data_reader = csv.DictReader(data_csv)
            for datapoint in data_reader:
                image, mask, pick_dot, place_dot, label = self.read_datapoint(datapoint)

                if self.dataset_type == 'train':
                    if label == 0:
                        self.samples.append([image, mask, pick_dot, place_dot, label])

                    else:
                        self.samples.append([image, mask, pick_dot, place_dot, label])
                        image_flipped = transforms.functional.hflip(image)
                        mask_flipped = transforms.functional.hflip(mask)
                        pick_dot_flipped = transforms.functional.hflip(pick_dot)
                        place_dot_flipped = transforms.functional.hflip(place_dot)
                        self.samples.append([image_flipped, mask_flipped, pick_dot_flipped, place_dot_flipped, label])

                else:
                    self.samples.append([image, mask, pick_dot, place_dot, label])

            print("All data have been loaded! Total dataset size: {:d}".format(len(self.samples)))
            # print(self.samples[1])
