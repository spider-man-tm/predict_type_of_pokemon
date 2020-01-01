import os
import cv2
import torch
from torch.utils.data import Dataset

from utils import return_type, type_to_array, load_typ_yml


type_dic = load_typ_yml()

class PokemonDataset(Dataset):
    def __init__(self, df, img_path, aug, transforms):
        self.df = df
        self.img_path = img_path
        self.aug = aug
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df['id'][idx]
        img = cv2.imread(os.path.join(self.img_path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.aug(image=img)
        img = augmented['image']
        img = img/255
        img = self.transforms(img)

        label = return_type(img_name, type_dic)
        label = type_to_array(label)
        label = torch.from_numpy(label)

        return img, label