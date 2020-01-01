import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

from aug import train_augmentation, val_augmentation, mixup
from config import Config
from dataset import PokemonDataset
from loss import criterion
from utils import seed_everything, metric, load_typ_yml


export_model = './export_model'
export_figure = './export_figure'
train_csv_path = 'data/train.csv'
train_img_path = 'data/train'

config = Config()

num_epochs = config.num_epochs
seed = config.seed
batch_size = config.batch_size
device = config.device
batch_multiplier = config.batch_multiplier
use_mixup = config.use_mixup

train_aug = train_augmentation()
val_aug = val_augmentation()
transforms = transforms.Compose([
    transforms.ToTensor(),
])
type_dic = load_typ_yml()

seed_everything(seed)
train = pd.read_csv(train_csv_path)
train, valid = train[:2680], train[2680:].reset_index(drop=True)

train_dataset = PokemonDataset(train, train_img_path, train_aug, transforms)
valid_dataset = PokemonDataset(valid, train_img_path, val_aug, transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
loaders_dict = {'train': train_loader, 'val': valid_loader}

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=18)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, cooldown=0)


def main():
    torch.backends.cudnn.benchmark = True
    num_train_imgs = len(loaders_dict['train'].dataset)
    num_val_imgs = len(loaders_dict['val'].dataset)
    batch_size = loaders_dict['train'].batch_size
    logs = []

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        epoch_train_score = 0.0
        epoch_val_score = 0.0

        print('-----------------------')
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-----------------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                optimizer.zero_grad()
            else:
                model.eval()

            count = 0
            for img_batch, label_batch in loaders_dict[phase]:
                if use_mixup:
                    mixup_flag = np.random.randint(use_mixup)==1
                    if mixup_flag:
                        img_batch, label_batch = mixup(img_batch, label_batch, alpha=1, n_classes=18)
                img_batch = img_batch.to(device, dtype=torch.float)
                label_batch = label_batch.to(device, dtype=torch.float)

                if (phase=='train') and (count==0):
                    optimizer.step()
                    optimizer.zero_grad()
                    count = batch_multiplier
                
                with torch.set_grad_enabled(phase == 'train'):
                    output = torch.sigmoid(model(img_batch))

                    if phase == 'train':
                        loss = criterion(output, label_batch)
                        loss /= batch_multiplier
                        loss.backward()
                        count -= 1
                        epoch_train_loss += loss.item() * batch_multiplier

                        for pred, label in zip(output, label_batch):
                            pred = pred.detach().cpu().numpy()
                            label = label.detach().cpu().numpy()
                            epoch_train_score += metric(label, pred)

                    else:
                        loss = criterion(output, label_batch)
                        loss /= batch_multiplier
                        epoch_val_loss += loss.item() * batch_multiplier

                        for pred, label in zip(output, label_batch):
                            pred = pred.detach().cpu().numpy()
                            label = label.detach().cpu().numpy()
                            epoch_val_score += metric(label, pred)

        train_loss = epoch_train_loss / num_train_imgs
        val_loss = epoch_val_loss / num_val_imgs
        train_score = epoch_train_score / num_train_imgs
        val_score = epoch_val_score / num_val_imgs

        t_epoch_finish = time.time()
        print(f'epoch: {epoch+1}')
        print(f'Epoch_Train_Loss: {train_loss:.3f}')
        print(f'Epoch_Val_Loss: {val_loss:.3f}\n')
        print(f'Epoch_Train_Score: {train_score:.3f}')
        print(f'Epoch_Val_Score: {val_score:.3f}\n')
        print('timer:  {:.3f} sec.'.format(t_epoch_finish - t_epoch_start), '\n')
        t_epoch_start = time.time()
        for g in optimizer.param_groups:
            print('lr: ', g['lr'], '\n\n')

        log_epoch = {
            'epoch': epoch+1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_score': train_score,
            'val_score': val_score,
            }
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv(f'{export_model}/log.csv', index=False)
        torch.save(model.state_dict(), f'{export_model}/model_epoch{epoch+1}.pth')

        scheduler.step(val_loss)

    df = pd.read_csv(f'{export_model}/log.csv')
    plt.plot(df['train_loss'], label='train loss')
    plt.plot(df['val_loss'], label='val loss')
    plt.legend()
    plt.savefig(f'{export_figure}/loss.png')
    plt.close()

    plt.plot(df['train_score'], label='train score')
    plt.plot(df['val_score'], label='val score')
    plt.legend()
    plt.savefig(f'{export_figure}/score.png')
    plt.close()

if __name__ == "__main__":
    main()