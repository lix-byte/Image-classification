import time

import torch
from torch import nn, optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from net import Model
from labelsmoothing import LabelSmoothing

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

train_path = r'./Brain Tumor Classification/train'
val_path = r'./Brain Tumor Classification/val'
lr = 0.01
epochs =5

train_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
        ]
    )

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ]
    )

train_data = datasets.ImageFolder(train_path,transform = train_transforms)
val_data = datasets.ImageFolder(val_path,transform = val_transforms)

classes = val_data.classes
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=2)
net = Model().cuda()

def train_val():

    train_datasize = len(train_data)
    val_datasize = len(val_data)
    print('Classes size : ', classes)
    print('Total train size : ', train_datasize)
    print('Total val size : ', val_datasize)

    cirterion = LabelSmoothing(smoothing=0.05)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    epoch_list = []
    net.train()
    loss_list = []
    acc_list = []
    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch+1,epochs))

        train_loss = 0.0
        train_count = 0
        val_loss = 0.0
        val_count = 0
        for i,(inputs,labels) in enumerate(tqdm(train_dataloader)):
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = cirterion(outputs,labels)
            pred = outputs.argmax(dim=1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_count += torch.sum(pred == labels.data)
        train_loss = train_loss / len(train_data.targets)
        train_acc = train_count.double() / len(train_data.targets)

        print('train loss : {:.4f} , train acc : {:.4f}'.format(train_loss,train_acc))

        with torch.no_grad():
            net.eval()

            for j,(inputs,labels) in enumerate(tqdm(val_dataloader)):
                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs = net(inputs)
                loss = cirterion(outputs, labels)
                pred = outputs.argmax(dim=1)
                val_loss += loss.item() * inputs.size(0)
                val_count += torch.sum(pred == labels.data)
            val_loss = val_loss / len(val_data.targets)
            val_acc = val_count.double() / len(val_data.targets)
            loss_list.append([train_loss,val_loss])
            acc_list.append([train_acc,val_acc])

            print('val loss : {:.4f} , val acc : {:.4f}'.format(val_loss, val_acc))


        epoch_list.append(epoch+1)
    torch.save(net, './models/{}_{:.4f}_model.pth'.format(epoch+1, val_acc))
    plt.figure(1)
    plt.plot(epoch_list,acc_list)
    plt.legend(['Train Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.ylim(0, 1)
    plt.savefig('./loss_acc/accuracy_curve.png')
    # plt.show()

    plt.figure(2)
    plt.plot(epoch_list,loss_list)
    plt.legend(['Train Loss', 'Val Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('./loss_acc/loss_curve.png')
    #plt.show()



if __name__ == "__main__":
    train_val()