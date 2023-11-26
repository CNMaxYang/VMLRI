import os
import sys
import dataSplit
import numpy as np
import pandas as pd
import models_vit
import sklearn.metrics as metrics
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler
from util.pos_embed import interpolate_pos_embed
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm



class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def balanceData(train_images_label):
    count = np.zeros(2)
    for i in train_images_label:
        if i == 0:
            count[0] += 1
        if i == 1:
            count[1] += 1
    classWeight = [count.sum() / count[j] for j in range(2)]
    return classWeight


def main(train_images_path, train_images_label, val_images_path, val_images_label):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.Resize((448, 448)),
                                     transforms.RandomRotation(180),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.486, 0.251, 0.079], [0.255, 0.142, 0.072])]),
        "val": transforms.Compose([transforms.Resize((448, 448)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.486, 0.251, 0.079], [0.255, 0.142, 0.072])])}
    train_dataset = MyDataSet(train_images_path, train_images_label, transform=data_transform["train"])
    train_num = train_dataset.__len__()

    batch_size = 24
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    weight = balanceData(train_images_label)
    samples_weight = torch.tensor([weight[t] for t in train_images_label])
    samper = WeightedRandomSampler(samples_weight, len(train_images_label))
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=samper,
                                               batch_size=batch_size,
                                               num_workers=nw)

    validate_dataset = MyDataSet(val_images_path, val_images_label, transform=data_transform["val"])
    val_num = validate_dataset.__len__()
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False, num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    model = models_vit.vit_base_patch16(
        num_classes=1,
        global_pool=False,
        img_size=448
    )
    checkpoint = torch.load('448bamde/100.pth', map_location='cpu')
    model_name = "448bamde100"
    dir_name = '448bamde100'
    os.makedirs(dir_name, exist_ok=True)

    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    model.to(device)
    print(model)

    lr = 1.5625e-3
    # lr = 1e-4
    gamma = 0.8
    milestones = [10, 25, 50]
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    loss_function = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    epochs = 150
    best_auc = 0.0
    best_labels = []
    best_prob = []
    train_steps = len(train_loader)
    dev_steps = len(validate_loader)
    saveTrainLoss = []
    saveDevLoss = []
    saveValidAcc = []
    saveValidSen = []
    saveValidSpe = []
    saveAUC = []
    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0.0
        dev_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            labels = labels.reshape((images.shape[0], 1)).float()
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.5f}".format(epoch + 1, epochs, loss)

        # validate
        model.eval()
        wholePre = []
        wholeLabel = []
        wholePreSigmod = []
        correct = 0
        total = 0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                loss_labels = val_labels
                loss_labels = loss_labels.reshape((val_images.shape[0], 1)).float()
                wholeLabel = np.append(wholeLabel, val_labels.detach().numpy())
                val_labels = val_labels.reshape((val_images.shape[0], 1)).float()
                outputs = model(val_images.to(device))
                predicted = torch.sigmoid(outputs)
                predicted = predicted.cpu()
                predicted = predicted.detach().numpy()
                wholePreSigmod = np.append(wholePreSigmod, predicted.flatten())
                predicted = np.where(predicted > 0.5, 1, 0)
                total += val_labels.size(0)
                val_labels = val_labels.detach().numpy()
                correct += (predicted == val_labels).sum()
                wholePre = np.append(wholePre, predicted.flatten())

                loss = loss_function(outputs, loss_labels.to(device))
                dev_loss += loss.item()

        scheduler.step()
        acc = 100 * correct / total
        c = metrics.confusion_matrix(wholeLabel, wholePre)

        # dispcm = metrics.ConfusionMatrixDisplay(confusion_matrix=c, display_labels=['0', '1'])
        # dispcm.plot()
        # plt.show()
        # fpr, tpr, thresholds = metrics.roc_curve(wholeLabel, wholePreSigmod)
        # roc_auc = metrics.auc(fpr, tpr)
        # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        # display.plot()
        # plt.show()

        sen = c[1][1] / (c[1][1] + c[1][0])
        spe = c[0][0] / (c[0][0] + c[0][1])
        auc = metrics.roc_auc_score(wholeLabel, wholePreSigmod)
        print('[epoch %d] train_loss: %.3f dev_loss: %.3f  val_accuracy: %.3f Sen: %.3f Spe: %.3f Auc: %.3f'
              % (epoch + 1, running_loss / train_steps, dev_loss / dev_steps, acc, sen, spe, auc))
        saveAUC = np.append(saveAUC, auc)
        saveTrainLoss = np.append(saveTrainLoss, running_loss / train_steps)
        saveDevLoss = np.append(saveDevLoss, dev_loss / dev_steps)
        saveValidAcc = np.append(saveValidAcc, acc)
        saveValidSen = np.append(saveValidSen, sen)
        saveValidSpe = np.append(saveValidSpe, spe)
        if auc > best_auc:
            best_auc = auc
            torch.save(model, dir_name + '/' + model_name)
            best_labels = wholeLabel
            best_labels = best_labels.reshape(-1, 1)
            best_prob = wholePreSigmod
            best_prob = best_prob.reshape(-1, 1)
    savedata = np.vstack((saveTrainLoss, saveDevLoss, saveValidAcc, saveAUC, saveValidSen, saveValidSpe))
    saveROC = np.hstack((best_labels, best_prob))
    savedata = pd.DataFrame(savedata)
    saveROC = pd.DataFrame(saveROC)
    writer = pd.ExcelWriter(dir_name + '/' + model_name + '.xlsx')
    savedata.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()
    writer = pd.ExcelWriter(dir_name + '/' + model_name + 'ROC.xlsx')
    saveROC.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()
    save_info = []
    save_info.extend([model_name, 'batch_size:', batch_size, 'SGD:', lr, gamma, milestones, best_auc])
    f = open(dir_name + '/' + 'info.txt', 'w')
    for i in save_info:
        f.write(str(i))
        f.write(' ')
    f.close()
    print('Best Auc: {:.5f}'.format(best_auc))
    print('Finished Training')


if __name__ == '__main__':
    train_images_path, train_images_label, val_images_path, val_images_label = \
        dataSplit.read_split_data('../data/binaryclassification', plot_image=False)
    main(train_images_path, train_images_label, val_images_path, val_images_label)
