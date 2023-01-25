#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet34, resnet18, resnet50
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
import math
import random
import copy
import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ThyDataset(Dataset):
    def __init__(self, csv1_path, csv2_path, transform=None):
        self.df = pd.read_csv(csv1_path)
        self.unique_patients = list(set(self.df.patient_name))
        self.unique_patients.sort(key=list(self.df.patient_name).index)
        self.ref_df = pd.read_csv(csv2_path)
        self.transform = transform
        self.patient = None

    def __len__(self):
        return len(self.unique_patients)

    def __getitem__(self, index):
        images = []
        self.patient = self.unique_patients[index]
        label = self.ref_df[self.ref_df['patient_name']== self.patient].histo_label.item()
        image_paths_of_this_patient = list(self.df.path[self.df.patient_name == self.patient])
        for image_path in image_paths_of_this_patient:
            image = Image.open(image_path)
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)
        images = torch.stack(images, dim=0)
        return images, label, self.patient

# build our own collate_fn
def pad_tensor(tensor, pad, dim):
    """
    tensor: tensor to pad
    pad: the size to pad to
    dim: dimension to pad
    """
    pad_size = list(tensor.shape)
    pad_size[dim] = pad - tensor.size(dim)
    return torch.cat([tensor, torch.zeros(pad_size)], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # pad according to max_len
        images = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        patients = [x[2] for x in batch]
        batch = map(lambda x, y, z: [pad_tensor(x, pad=max_len, dim=self.dim), y, z], images, labels, patients)
        # stack all
        batch = list(batch)
        xs = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
        ys = torch.LongTensor(list(map(lambda x: x[1], batch)))
        zs = list(map(lambda x: x[2], batch))
        return xs, ys, zs
    
    def __call__(self, batch):
        return self.pad_collate(batch)

"""
IRAM
args:
    C1: channel of feature extracted by feature extractor, equals to C
    C2: channel C'
    dropout: whether to use dropout (p=0.25)
    tau: scaling factor for psi12
"""
class IRAM(nn.Module):
    
    def __init__(self, C1, C2=128, dropout=True, tau=128):
        super(IRAM, self).__init__()
        self.C1 = C1
        self.C2 = C2
        self.psi1 = [
            nn.Conv2d(C1, C2, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        ]
        self.psi2 = [
            nn.Conv2d(C1, C2, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        ]
        self.psi3 = [
            nn.Conv2d(C1, C2, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        ]
        self.psi4 = [
            nn.Linear(C2, C1),
            nn.ReLU()
        ]
        if dropout:
            self.psi1.append(nn.Dropout(0.25))
            self.psi2.append(nn.Dropout(0.25))
            self.psi3.append(nn.Dropout(0.25))
            self.psi4.append(nn.Dropout(0.25))
        self.tau = tau
        self.psi1 = nn.Sequential(*self.psi1)
        self.psi2 = nn.Sequential(*self.psi2)
        self.psi3 = nn.Sequential(*self.psi3)
        self.psi4 = nn.Sequential(*self.psi4)
        
    def forward(self, x):
        x_psi1 = self.psi1(x)
        x_psi1 = x_psi1.view(-1, self.C2)
        x_ds = F.interpolate(x, scale_factor=0.5)
        x_psi2 = self.psi2(x_ds)
        x_psi2 = x_psi2.view(self.C2, -1)
        x_psi3 = self.psi3(x_ds)
        x_psi3 = x_psi3.view(-1, self.C2)
        psi12 = F.softmax(torch.mm(x_psi1, x_psi2) / self.tau)
        psi123 = torch.mm(psi12, x_psi3)
        psi1234 = self.psi4(psi123)
        M = psi1234.view(x.shape) + x
        return M, x


"""
ISAM
args:
    batch_size: batch size
    L: input feature dimension of M, equals to C1 in IRAM module
    D: hidden layer dimension
    dropout: whether to use dropout (p=0.25)
    n_classes: number of classes
"""
class ISAM(nn.Module):
    
    def __init__(self, batch_size, L, D=256, n_classes=1, dropout=True):
        super(ISAM, self).__init__()
        self.L = L
        self.D = D
        self.batch_size = batch_size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()
        ]
        self.attention_b = [
            nn.Linear(L, D),
            nn.Sigmoid()
        ]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
            
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)
        
    def forward(self, x):
        x_pooled = self.pool(x).view(x.shape[0], self.L)
        a = self.attention_a(x_pooled)
        b = self.attention_b(x_pooled)
        A = a.mul(b)
        A = self.attention_c(A)
        if not self.training:
            A = A.view(1, 1, -1)
        else:
            A = A.view(self.batch_size, 1, -1)
        return A, x_pooled, x


"""
Args:
    C2: channel C', default 128
    D: hidden layer dimension, default 256
    batch_size: batch_size
    num_cls: number of classes, default 2
    n_channels: image channel, 3 in this study
    dropout: whether to use dropout (p=0.25)
"""
class ThyNet(nn.Module):
    
    def __init__(self, C2, D, batch_size, dropout=True, num_cls=2, n_channels=3):
        super(ThyNet, self).__init__()
        feature_extractor = resnet34(pretrained=True)
        # feature_extractor = resnet18(pretrained=True)
        #res34
        self.feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-2])
        #for p in self.feature_extractor.parameters():
            #p.requires_grad = False
        self.C1 = self.feature_extractor[-1][-1].conv2.out_channels
        self.L = self.C1
        self.iram = IRAM(C1=self.C1, C2=C2, dropout=dropout, tau=C2)
        self.isam = ISAM(batch_size=batch_size, L=self.L, D=D, dropout=dropout)
        self.classifier = nn.Linear(self.C1, num_cls)
        
        self.n_channels = n_channels
        self.batch_size = batch_size
        
    def forward(self, x):
        import ipdb;ipdb.set_trace()
        ori = x
        x = x.view(-1, self.n_channels, x.shape[-2], x.shape[-1])
        feat = self.feature_extractor(x)
        M, _ = self.iram(feat)
        A, x_pooled, _ = self.isam(M)
        A = F.softmax(A, dim=2)  # softmax over instances
        if not self.training:
            x_pooled = x_pooled.view(1, -1, self.C1)
            h = torch.bmm(A, x_pooled).view(1, self.C1)
        else:
            x_pooled = x_pooled.view(self.batch_size, -1, self.C1)
            h = torch.bmm(A, x_pooled).view(self.batch_size, self.C1)
        logits = self.classifier(h)
        return logits


class ThyNet2(nn.Module):
    
    def __init__(self, batch_size, num_cls=2, n_channels=3):
        super(ThyNet2, self).__init__()
        #feature_extractor = resnet34(pretrained=True)
        feature_extractor = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-2])
        #for p in self.feature_extractor.parameters():
            #p.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_cls)
        
        self.n_channels = n_channels
        self.batch_size = batch_size
        
    def forward(self, x):
        x = x.view(-1, self.n_channels, x.shape[-2], x.shape[-1])
        feat = self.feature_extractor(x)
        feat = self.pool(feat).reshape(feat.shape[0], 512)
        if self.training:
            feat = feat.view(self.batch_size, -1, 512)
        else:
            feat = feat.view(1, -1, 512)
        feat, _ = torch.max(feat, 1)
        logits = self.classifier(feat)
        return logits

class ThyNet3(nn.Module):
    
    def __init__(self, batch_size, num_cls=2, n_channels=3):
        super(ThyNet3, self).__init__()
        feature_extractor = resnet34(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-2])
        #for p in self.feature_extractor.parameters():
            #p.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_cls)
        
        self.n_channels = n_channels
        self.batch_size = batch_size
        
    def forward(self, x):
        x = x.view(-1, self.n_channels, x.shape[-2], x.shape[-1])
        feat = self.feature_extractor(x)
        feat = self.pool(feat).reshape(feat.shape[0], 512)
        if self.training:
            feat = feat.view(self.batch_size, -1, 512)
        else:
            feat = feat.view(1, -1, 512)
        feat = torch.mean(feat, 1)
        logits = self.classifier(feat)
        return logits
"""
Args:
    csv1_path: image_paths.csv in ./tmp folder
    csv2_path: patient_name_label.csv in ./tmp folder
    test_size: size of test set for external testing
    seed: random seed
    k: number of folds
    split_dir: dir to save splitted csv files, default ./split folder
"""
def split(csv1_path, csv2_path, test_size=0.1, seed=202203, k=5, split_dir="./split"):
    print("Splitting %d fold csv files..." % k)
    os.makedirs(split_dir, exist_ok=True)
    all_image_paths_df = pd.read_csv(csv1_path)
    all_patient_and_labels_df = pd.read_csv(csv2_path)
    train_val_patient_df, test_patient_df = train_test_split(all_patient_and_labels_df,
                                                             test_size=test_size,
                                                             random_state=seed)
    print("Saving patient level test df")
    test_patient_df.to_csv(os.path.join(split_dir, "test_case.csv"), index=False)
    test_images_paths_df = all_image_paths_df[
        all_image_paths_df["patient_name"].isin(test_patient_df["patient_name"])
    ]
    print("Saving image level test df")
    test_images_paths_df.to_csv(os.path.join(split_dir, "test_image.csv"), index=False)
    kf = KFold(n_splits=k, random_state=seed, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(train_val_patient_df)):
        train_patient_df, val_patient_df = train_val_patient_df.iloc[train_index], train_val_patient_df.iloc[test_index]
        print("Saving patient level df fold %d" % i)
        train_patient_df.to_csv(os.path.join(split_dir, "train_case_split_" + str(i) + ".csv"), index=False)
        val_patient_df.to_csv(os.path.join(split_dir, "val_case_split_" + str(i) + ".csv"), index=False)
        train_image_paths_df = all_image_paths_df[
            all_image_paths_df["patient_name"].isin(train_patient_df["patient_name"])
        ]
        val_image_paths_df = all_image_paths_df[
            all_image_paths_df["patient_name"].isin(val_patient_df["patient_name"])
        ]
        print("Saving image level df fold %d" % i)
        train_image_paths_df.to_csv(os.path.join(split_dir, "train_image_split_" + str(i) + ".csv"), index=False)
        val_image_paths_df.to_csv(os.path.join(split_dir, "val_image_split_" + str(i) + ".csv"), index=False)


"""
Learning rate warming up
"""
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """learning rate warmup"""

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


@torch.no_grad()
def eval_process(epoch, model, criterion, dataloader, device):
    print("Epoch %d Validation......" % epoch)
    cpu_device = torch.device("cpu")
    model.eval()
    labels = []
    preds = []
    running_loss = 0.
    for images, targets, _ in tqdm(dataloader):
        images = images.to(device)
        labels.extend(targets.tolist())
        targets = targets.to(device)
        logits = model(images)
        preds.extend(logits.cpu().numpy()[:, -1].tolist())
        loss = criterion(logits, targets)
        running_loss += loss.item() * images.size(0)

    eval_loss = running_loss / len(dataloader.dataset)
    print("Eval val loss: %.4f" % eval_loss)
    
    auc_score = roc_auc_score(labels, preds)
    print("Val AUC: %.4f" % auc_score)
    return auc_score, eval_loss


@torch.no_grad()
def test_process(model, criterion, dataloader, device):
    cpu_device = torch.device("cpu")
    model.eval()
    labels = []
    preds_logits = []
    preds_probs = []
    running_loss = 0.
    all_patient_names = []
    for ind, (images, targets, patient_names) in enumerate(tqdm(dataloader)):
        import ipdb;ipdb.set_trace()
        images = images.to(device)
        labels.extend(targets.tolist())
        targets = targets.to(device)
        logits = model(images)
        pred_probs = torch.softmax(logits, 1)
        preds_probs.extend(pred_probs.cpu().numpy()[:, -1].tolist())
        loss = criterion(logits, targets)
        running_loss += loss.item() * images.size(0)
        all_patient_names.extend(patient_names)
        preds_logits.extend(logits.cpu().numpy()[:, -1].tolist())

    test_loss = running_loss / len(dataloader.dataset)
    print("Test loss: %.4f" % test_loss)
    
    auc_score = roc_auc_score(labels, preds_logits)
    average_precision = average_precision_score(labels, preds_logits)
    print("Test AUC: %.4f" % auc_score)
    print("Test AP: %.4f" % average_precision)
    # plot_roc_curve(labels, final_logits)
    # plot_pr_curve(labels, final_logits)
    df = pd.DataFrame({"patient_name": all_patient_names,
                       "prob": preds_probs,
                       "logit": preds_logits,
                       "label": labels})

    return auc_score, test_loss, df


def train_process(model, criterion, optimizer, lr_sche, dataloaders,
                  num_epochs, use_tensorboard, device,
                  save_model_path, record_iter, writer=None):
    model.train()

    best_score = 0.0
    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        lr_scheduler = None
        running_loss = 0.0
        print("====Epoch{0}====".format(epoch))
        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(dataloaders["train"]) - 1)
            lr_scheduler = warmup_lr_scheduler(
                optimizer, warmup_iters, warmup_factor
            )

        for i, (images, targets, _) in enumerate(tqdm(dataloaders["train"])):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            running_loss += loss.item() * images.size(0)

            lr = optimizer.param_groups[0]["lr"]

            if (i + 1) % record_iter == 0:
                to_date_cases = (i + 1) * images.size(0)
                tmp_loss = running_loss / to_date_cases
                print("Epoch{0} loss:{1:.4f}".format(epoch, tmp_loss))
                
                if use_tensorboard:
                    writer.add_scalar("Train loss",
                                      tmp_loss,
                                      epoch * len(dataloaders["train"]) + i)
                    writer.add_scalar("lr", lr,
                                      epoch * len(dataloaders["train"]) + i)

        val_auc, val_loss = eval_process(
            epoch, model, criterion, dataloaders["val"], device
        )
        if lr_sche is not None:
            lr_sche.step()

        if val_auc > best_score:
            best_score = val_auc
            best_state_dict = copy.deepcopy(model.state_dict())

        if use_tensorboard:
            writer.add_scalar(
                "validataion AUC", val_auc, global_step=epoch
            )
            writer.add_scalar(
                "validation loss", val_loss, global_step=epoch
            )

        model.train()

    print("Training Done!")
    print("Best Valid AUC: %.4f" % best_score)
    torch.save(best_state_dict, save_model_path)

    print("========Start Testing========")
    model.load_state_dict(best_state_dict)
    test_auc, test_loss, df = test_process(
        model, criterion, dataloaders["test"], device
    )
    if use_tensorboard:
        writer.add_scalar("Test AUC", test_auc, global_step=0)
        writer.close()


"""
Cutout: Randomly mask out one or more patches from an image
        ref paper: Improved Regularization of Convolutional Neural Networks with Cutout.
"""
class Cutout(object):
    """Randomly mask out one or more patches from an image"""
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (tensor): Tensor image of size (C, H, W)
        Returns:
            Image with n_holes of dimension lengthxlength cut out of it
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


"""
RandomRotation
"""
class MyRotationTrans:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return torchvision.transforms.functional.rotate(x, angle)


# Build the final transform
#transform = transforms.Compose([
#        transforms.Resize((256, 256)),
#        transforms.RandomHorizontalFlip(p=0.5),
#        MyRotationTrans([0, 90, 180, 270]),
#        transforms.ColorJitter(),
#        #transforms.RandomVerticalFlip(),
#        transforms.ToTensor(),
#        Cutout(1, 100),
#        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#    ])

#val_transform = transforms.Compose([
#        transforms.Resize((256, 256)),
#        transforms.ToTensor(),
#        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#    ])
transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# function for k-fold cross validation training procedure
def train_thynet(k, split_dir, save_model_dir,
                 train_trans, val_trans, batch_size,
                 num_workers, C2, D, device, lr,
                 momentum, weight_decay, gamma,
                 logdir, num_epochs, use_tensorboard,
                 record_iter):
    test_csv1_path = os.path.join(split_dir, "test_image.csv")
    test_csv2_path = os.path.join(split_dir, "test_case.csv")
    print("Building test dataset...")
    test_dataset = ThyDataset(csv1_path=test_csv1_path, csv2_path=test_csv2_path,
                              transform=val_trans)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
    #                          num_workers=num_workers)
    for i in range(k):
        print("====Starting Fold %d====" % (i + 1))
        print("Building fold %d train dataset..." % (i + 1))
        train_csv1_path = os.path.join(split_dir, "train_image_split_" + str(i) + ".csv")
        train_csv2_path = os.path.join(split_dir, "train_case_split_" + str(i) + ".csv")
        train_dataset = ThyDataset(csv1_path=train_csv1_path, csv2_path=train_csv2_path,
                                   transform=train_trans)
        print("Building fold %d val dataset..." % (i + 1))
        val_csv1_path = os.path.join(split_dir, "val_image_split_" + str(i) + ".csv")
        val_csv2_path = os.path.join(split_dir, "val_case_split_" + str(i) + ".csv")
        val_dataset = ThyDataset(csv1_path=val_csv1_path, csv2_path=val_csv2_path,
                                 transform=val_trans)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, collate_fn=PadCollate(dim=0),
                                  num_workers=num_workers, drop_last=True)
        # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
        #                         num_workers=num_workers)
        val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                num_workers=num_workers)
        test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                num_workers=num_workers)
        dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
        print("Dataset Done...")
        
        print("Preparing Model...")
        # thynet = ThyNet(C2=C2, D=D, batch_size=batch_size)
        # thynet = thynet.to(device)
        thynet = ThyNet2(batch_size=batch_size)
        # thynet = ThyNet3(batch_size=batch_size)
        thynet = thynet.to(device)
        #if torch.cuda.device_count() > 1:
            #thynet = nn.DataParallel(thynet)
        print("Model Done...")
        
        #params = filter(lambda p: p.requires_grad, thynet.parameters())
        params = [p for p in thynet.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay)
        #optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75],
                                                            gamma=gamma)
        criterion = nn.CrossEntropyLoss()
        print("Start Training...")
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)
        os.makedirs(save_model_dir, exist_ok=True)
        save_model_path = os.path.join(save_model_dir, "thynet_fold_%d.pth" % (i + 1))
        train_process(model=thynet, criterion=criterion, optimizer=optimizer,
                      lr_sche=lr_scheduler, dataloaders=dataloaders, writer=writer,
                      num_epochs=num_epochs, use_tensorboard=use_tensorboard,
                      device=device, save_model_path=save_model_path,
                      record_iter=record_iter)
        
    print("====Training Done====")


if __name__ == "__main__":
    csv1_path = "./tmp/image_paths.csv"
    csv2_path = "./tmp/patient_name_label.csv"
    #split(csv1_path, csv2_path)

    # settings
    k = 5
    split_dir = "./split/"
    save_model_dir = "./result/res18_max/"
    train_trans = transform
    val_trans = val_transform
    batch_size = 10
    num_workers = 12
    C2 = 128
    D = 256
    device = device
    lr = 0.001
    momentum = 0.9
    weight_decay = 5e-4
    gamma = 0.1
    logdir = "./logs/"
    num_epochs = 100
    use_tensorboard = True
    record_iter = 10

    # train_thynet(k, split_dir, save_model_dir,
    #              train_trans, val_trans, batch_size,
    #              num_workers, C2, D, device, lr,
    #              momentum, weight_decay, gamma,
    #              logdir, num_epochs, use_tensorboard,
    #              record_iter)
    model_path = "./result/thynet_fold_1.pth"
    # model_path = "./result/exp_min/thynet_fold_5.pth"
    # # model_path = "./result/exp_new/thynet_fold_1.pth"
    # # model_path = "./result/exp_max/thynet_fold_5.pth"
    thynet = ThyNet(C2=C2, D=D, batch_size=1)
    # # thynet = ThyNet2(batch_size=1)
    # thynet = ThyNet3(batch_size=1)
    thynet = thynet.to(device)
    thynet.is_training = False
    # thynet.load_state_dict(torch.load(model_path))
    criterion = nn.CrossEntropyLoss()
    # # test_dataset = ThyDataset(csv1_path="./split/test_image.csv", csv2_path="./split/test_case.csv",
    # #                                                    transform=val_trans)
    # test_dataset = ThyDataset(csv1_path="./split/tmp_image_dingxing.csv", csv2_path="./split/tmp_case_dingxing.csv",
    #                                                    transform=val_trans)
    test_dataset = ThyDataset(csv1_path="./split/tmp_image_dingxing2.csv", csv2_path="./split/tmp_case_dingxing2.csv",
                                                       transform=val_trans)
    # test_dataset = ThyDataset(csv1_path="./split/tmp_image.csv", csv2_path="./split/tmp_case.csv",
    #                                                    transform=val_trans)
    # # test_dataset = ThyDataset(csv1_path="./split/val_image_split_0.csv", csv2_path="./split/val_case_split_0.csv",
    # #                                                    transform=val_trans)
    # test_dataset = ThyDataset(csv1_path="./data/external_image_new.csv", csv2_path="./data/external_case.csv",
    #                                                    transform=val_trans)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                     num_workers=num_workers)
    test_auc, test_loss, df = test_process(
            thynet, criterion, test_loader, device
        
    )
    # df.to_csv("./result/external_fold5_mean.csv", index=False)
