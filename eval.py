import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

from aug import val_augmentation
from config import Config
from dataset import PokemonDataset
from utils import (seed_everything,
                metric,
                load_typ_yml,
                predict_to_binary,
                confusion_df)


model_path = './export_model/model_epoch2.pth'
export_figure = './export_figure'
pred_path = './export_pred'
test_csv_path = 'data/test.csv'
test_img_path = 'data/test'

config = Config()

seed = config.seed
batch_size = config.batch_size
device = config.device

val_aug = val_augmentation()
transforms = transforms.Compose([
    transforms.ToTensor(),
])
type_dic = load_typ_yml()

seed_everything(seed)
test = pd.read_csv(test_csv_path)
test_dataset = PokemonDataset(test, test_img_path, val_aug, transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=18)
model.load_state_dict(torch.load(f'{model_path}'))
model.to(device)


def main():
    model.eval()
    predict = []
    score = 0
    for img_batch, label_batch in test_loader:
        img_batch = img_batch.to(device, dtype=torch.float)
        output = torch.sigmoid(model(img_batch))
        
        for pred, label in zip(output, label_batch):
            pred = pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            score += metric(label, pred)
            predict.append(pred)

    score /= len(test)
    print('Score: ', score)
    predict = pd.DataFrame(predict)
    predict_binary = predict_to_binary(test, predict)
    confusion = confusion_df(test, predict_binary)

    predict.to_csv(f'{pred_path}/predict.csv', index=False)
    predict_binary.to_csv(f'{pred_path}/predict_binary.csv', index=False)
    confusion.to_csv(f'{pred_path}/confusion.csv')

    plt.figure(figsize=(14, 4))
    plt.bar(x=[c for c in range(1, 19)], height=confusion.loc['F1_score', :], tick_label=confusion.columns, label='F1 Score')
    plt.legend()
    plt.savefig(f'{export_figure}/f1_score.png')


if __name__ == "__main__":
    main()