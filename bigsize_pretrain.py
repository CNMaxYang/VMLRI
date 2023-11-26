import os
import sys
import json
import models_mae
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from util import misc



class MyDataSet(Dataset):
    def __init__(self, images_path: list, transform=None):
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if self.transform is not None:
            img = self.transform(img)
        return img


def main(path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.Resize((320, 320)),
                                     transforms.RandomRotation(180),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.486, 0.251, 0.079], [0.255, 0.142, 0.072])])}
    file_list = []
    for i in os.listdir(path):
        file_list.append(os.path.join(path, i))
    train_dataset = MyDataSet(file_list, transform=data_transform["train"])

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                               batch_size=batch_size, pin_memory=False,
                                               num_workers=nw)


    model_name = "bamde32075"
    dir_name = 'BAMDE32075'
    os.makedirs(dir_name, exist_ok=True)
    net = models_mae.mae_vit_base_patch16_dec512d8b(img_size=320)
    print(net)
    net.to(device)
    # lr = 1.5625e-3
    lr = 1e-4
    # gamma = 0.8
    # milestones = [10, 40, 100]
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    epochs = 1000
    start_epoch = 0
    check_flag = False
    if check_flag:
        check_point = torch.load('checkpoint/400.pth', map_location=device)
        net.load_state_dict(check_point['model'])
        optimizer.load_state_dict(check_point['optimizer'])
        start_epoch = check_point['epoch']

    train_steps = len(train_loader)
    saveTrainLoss = []
    for epoch in range(start_epoch + 1, epochs + 1):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images = data
            optimizer.zero_grad()
            loss, _, _ = net(images.to(device), mask_ratio=0.75)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.5f}".format(epoch, epochs, loss)
        print(running_loss / train_steps)
        saveTrainLoss = np.append(saveTrainLoss, running_loss / train_steps)

        if epoch % 100 == 0:
            to_save = {
                'model': net.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            model_save_name = dir_name + '/' + str(epoch) + '.pth'
            misc.save_on_master(to_save, model_save_name)

    savedata = saveTrainLoss
    savedata = pd.DataFrame(savedata)

    writer = pd.ExcelWriter(dir_name + '/' + model_name + '.xlsx')
    savedata.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()
    save_info = []
    save_info.extend([model_name, 'batch_size:', batch_size, '固定学习率:', lr, 'Adam'])
    f = open(dir_name + '/' + 'info.txt', 'w')
    for i in save_info:
        f.write(str(i))
        f.write(' ')
    f.close()
    print('Finished Training')


if __name__ == '__main__':
    path = '../pretraindata'
    main(path)
