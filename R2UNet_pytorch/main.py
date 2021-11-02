import torch
import torch.nn as nn
import torch.optim as optim
from model import R2UNet, UNet, IterNet
import torchvision
from PIL import Image
import os
import numpy as np
import random
from tqdm import tqdm
from sklearn.feature_extraction import image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse
import paddle
from reprod_log import ReprodLogger
torch.set_printoptions(precision=8)
import collections
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


trans_fn1 = transforms.Compose([
    # transforms.CenterCrop(512),
    # transforms.RandomRotation(180),
    transforms.ToTensor(),
])

#random image transformation to do image augmentation
trans_fn2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

class UNetDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transforms = transform
        seed = random.randint(0, 2**32)
        self.imgs_path = list(sorted(os.listdir(os.path.join(self.root,"images"))))
        self.imgs = []
        for path in self.imgs_path:
            img_path = os.path.join(self.root, "images", path)
            img = Image.open(img_path)
            # image augmentation
            random.seed(seed)
            img = self.transforms(img)

            # randomly select patches from 20 training images
            img = torch.permute(img, [2, 1, 0])
            img = torch.from_numpy(image.extract_patches_2d(img, (48, 48), max_patches=1000, random_state=1))
            img = torch.permute(img, [0, 3, 1, 2])

            for i, sub_img in enumerate(img):
                self.imgs.append(sub_img)

        self.masks_path = list(sorted(os.listdir(os.path.join(self.root, "mask"))))
        self.masks = []
        for path in self.masks_path:
            mask_path = os.path.join(self.root, "mask", path)
            mask = Image.open(mask_path)
            random.seed(seed)
            mask = self.transforms(mask)

            mask = torch.permute(mask, [2, 1, 0])
            mask = torch.from_numpy(image.extract_patches_2d(mask, (48, 48), max_patches=1000, random_state=1))
            mask = mask.unsqueeze(3)
            mask = torch.permute(mask, [0, 3, 1, 2])

            for i, sub_mask in enumerate(mask):
                self.masks.append(sub_mask)

        self.targets_path =  list(sorted(os.listdir(os.path.join(self.root, "1st_manual"))))
        self.targets = []
        for path in self.targets_path:
            target_path = os.path.join(self.root, "1st_manual", path)
            target = Image.open(target_path)
            random.seed(seed)
            target = self.transforms(target)

            target = torch.permute(target, [2, 1, 0])
            target = torch.from_numpy(image.extract_patches_2d(target, (48, 48), max_patches=1000, random_state=1))
            target = target.unsqueeze(3)
            target = torch.permute(target, [0, 3, 1, 2])

            for i, sub_target in enumerate(target):
                self.targets.append(sub_target)


    def __getitem__(self, idx):
        
        # seed so image and target have the same random tranform
        seed = random.randint(0, 2**32)

        img = self.imgs[idx]
        random.seed(seed)
        # img = trans_fn2(img)


        mask = self.masks[idx]
        random.seed(seed)
        # mask = trans_fn2(mask)

        target = self.targets[idx]
        random.seed(seed)
        # target = trans_fn2(target)

        # fig = plt.figure()
        # ax1 = fig.add_subplot(1,3,1)
        # ax1.imshow(img.permute((1,2,0)))
        # ax2 = fig.add_subplot(1,3,2)
        # ax2.imshow(torch.squeeze(mask), cmap="gray")
        # ax3 = fig.add_subplot(1,3,3)
        # ax3.imshow(torch.squeeze(target), cmap="gray")
        # plt.show()


        return img.to(device), mask.to(device), target.to(device)

    def __len__(self):

        return len(self.imgs)


class model:

    def __init__(self, args):

        self.model = args.model
        self.epoch = args.epoch
        self.lr = args.lr
        self.batch_s = args.batch_size
        self.data_path = args.dataset_path
        self.output = args.result_path

        if self.model == 'U-Net':
            self.network = UNet()
        elif self.model == 'R2U-Net':
            self.network = R2UNet()
        elif self.model == 'IterNet':
            self.network = IterNet()
            
    def train(self):
        self.network.to(device)
        self.network.train()

        optimizer = optim.Adam(lr=self.lr, params=self.network.parameters())
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epoch, eta_min=0.00001)
        loss_fn = nn.BCELoss()

        for i in range(self.epoch):
            print('{} epoch {} {}'.format('=' * 10, i, '=' * 10))
            sum_loss = 0
            training_set = UNetDataset(self.data_path + 'training', trans_fn1)
            validation_set = UNetDataset(self.data_path + 'validation', trans_fn1)
            training_loader = DataLoader(training_set, batch_size=self.batch_s, shuffle=True)
            validation_loader = DataLoader(validation_set, batch_size=self.batch_s, shuffle=True)

            for img, mask, target in tqdm(training_loader):

                predict = self.network(img)

                if self.model == 'IterNet':
                    mask = mask.view(mask.size(0), -1)
                    target = target.view(target.size(0), -1)
                    loss = 0
                    for j in range(3):
                        iter_predict = predict[j].view(predict[j].size(0), -1)
                        iter_predict = iter_predict * mask.view(mask.size(0), -1)

                        loss += loss_fn(iter_predict, target)

                    sum_loss += loss.item()
                else:
                    predict = predict.view(predict.size(0), -1)
                    predict = predict * mask.view(mask.size(0), -1)
                    target = target.view(target.size(0), -1)


                    loss = loss_fn(predict, target)
                    sum_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

            sum_loss /= 230
            print('loss: {}'.format(sum_loss))

            if i % 5 == 0:
                torch.save(self.network.state_dict(), '{}{}{}.pth'.format(self.output, self.model, i))
            # Validation#
            results = []
            self.network.eval()
            with torch.no_grad():
                for img, mask, target in tqdm(validation_loader):
                    predict = self.network(img)
                    if self.model == 'IterNet':
                        predict = predict[-1]

                    predict = predict * mask
                    predict = (predict >= 0.5).astype(np.uint8)
                    target = target * mask
                    target = target.astype(np.uint8)
                    predict = np.array(predict)
                    target = np.array(target)

                    TP = np.sum(np.logical_and(predict == 1, target == 1))  # true positive
                    TN = np.sum(np.logical_and(predict == 0, target == 0))  # true negative
                    FP = np.sum(np.logical_and(predict == 1, target == 0))  # false positive
                    FN = np.sum(np.logical_and(predict == 0, target == 1))  # false negative

                    AC = (TP + TN) / (TP + TN + FP + FN)  # accuracy
                    SE = (TP) / (TP + FN)  # sensitivity
                    SP = TN / (TN + FP)  # specificity
                    precision = TP / (TP + FP)
                    recall = TP / (TP + FN)
                    F1 = 2 * ((precision * recall) / (precision + recall))
                    fpr, tpr, _ = roc_curve(target.flatten(), predict.flatten())
                    AUC = auc(fpr, tpr)
                    results.append((F1, SE, SP, AC, AUC))
                F1, SE, SP, AC, AUC = map(list, zip(*results))
                print(f'[Validation] F1:{sum(F1) / len(F1):.4f}, SE:{sum(SE) / len(SE):.4f}, SP:{sum(SP) / len(SP):.4f}, AC: {sum(AC) / len(AC):.4f}, AUC : {sum(AUC) / len(AUC):.4f}')

        torch.save(self.network.state_dict(), '{}{}.pth'.format(self.output,self.model))

    
    def test(self, show):
        '''
        run test set
        '''
        # load saved model
        self.network.to('cpu')
        self.network.load_state_dict(torch.load('{}{}.pth'.format(self.output,self.model), map_location=torch.device('cpu')))
        self.network.eval()

        # load test set
        self.imgs_path = list(sorted(os.listdir(self.data_path + 'testing/images')))
        self.masks_path = list(sorted(os.listdir(self.data_path + 'testing/mask')))
        self.targets_path = list(sorted(os.listdir(self.data_path + 'testing/1st_manual')))

        results = []
        with torch.no_grad():
            for img_name, mask_name, target_name in zip(self.imgs_path, self.masks_path, self.targets_path):
                img_path = os.path.join(self.data_path+'testing/images', img_name)
                img = Image.open(img_path)
                img = transforms.functional.center_crop(img, 560)
                img = transforms.functional.to_tensor(img).unsqueeze(0)

                mask_path = os.path.join(self.data_path+'testing/mask', mask_name)
                mask = Image.open(mask_path)
                mask = transforms.functional.center_crop(mask, 560)
                mask = np.array(mask).flatten() / 255
                mask = mask.astype(np.uint8)


                target_path = os.path.join(self.data_path+'testing/1st_manual', target_name)
                target = Image.open(target_path)
                target = transforms.functional.center_crop(target, 560)
                target = np.array(target)
                target_ = target.flatten() / 255
                target_ = target_.astype(np.uint8)
                target_ = target_[mask==1]

                predict = self.network(img)
                if self.model == 'IterNet':
                    predict = predict[-1]
                predict = np.squeeze(predict.numpy(), axis=(0,1))
                predict_ = predict.flatten()[mask==1]
                predict_ = (predict_>=0.5).astype(np.uint8)


                TP = np.sum(np.logical_and(predict_ == 1, target_ == 1)) # true positive
                TN = np.sum(np.logical_and(predict_ == 0, target_ == 0)) # true negative
                FP = np.sum(np.logical_and(predict_ == 1, target_ == 0)) # false positive
                FN = np.sum(np.logical_and(predict_ == 0, target_ == 1)) # false negative

                AC = (TP+TN)/(TP+TN+FP+FN) # accuracy
                SE = (TP)/(TP+FN) # sensitivity
                SP = TN/(TN+FP) # specificity
                precision = TP/(TP+FP)
                recall = TP/(TP+FN)
                F1 = 2*((precision*recall)/(precision+recall))
                fpr, tpr, _ = roc_curve(target_, predict_)
                AUC = auc(fpr,tpr)
                results.append((F1, SE, SP, AC, AUC))

                # show predicted image
                if show:
                    fig = plt.figure()
                    ax1 = fig.add_subplot(1,3,1)
                    ax1.imshow(img.squeeze(0).permute((1,2,0)))
                    ax2 = fig.add_subplot(1,3,2)
                    ax2.imshow(predict, cmap="gray")
                    ax3 = fig.add_subplot(1,3,3)
                    ax3.imshow(target, cmap="gray")
                    plt.show()

        F1, SE, SP, AC, AUC = map(list, zip(*results))

        print('F1 score: %.4f' %(sum(F1)/len(F1)))
        print('sensitivity: %.4f' %(sum(SE)/len(SE)))
        print('specificity: %.4f' %(sum(SP)/len(SP)))
        print('accuracy: %.4f' %(sum(AC)/len(AC)))
        print('AUC: %.4f' %(sum(AUC)/len(AUC)))

    def save_model(self):
        torch.save(self.network.state_dict(), 'model_pytorch.pth')

    def show_pkl(self):
        path_pytorch = "./R2U-Net.pth"
        torch_dict = torch.load(path_pytorch, map_location=torch.device('cpu'))
        for key in torch_dict:
            print(key)

    def pytorch2paddle(self):
        input_fp = "./R2U-Net.pth"
        output_fp = "../R2UNet_paddle/R2U-Net.pdparams"
        torch_dict = torch.load(input_fp)
        paddle_dict = {}
        for key in torch_dict:
            weight = torch_dict[key].cpu().detach().numpy()
            if 'running_mean' in key:
                key = key.replace('running_mean', '_mean')
            elif 'running_var' in key:
                key = key.replace('running_var', '_variance')
            paddle_dict[key] = weight

        paddle.save(paddle_dict, output_fp)

    def forward_pytorch(self):
        np.random.seed(0)
        torch.manual_seed(0)
        reprod_logger = ReprodLogger()
        self.network.load_state_dict(torch.load("./R2U-Net.pth"))
        self.network.cuda()
        self.network.eval()
        # read or gen fake data
        fake_data = np.load("../fake_data.npy")
        fake_data = torch.from_numpy(fake_data).cuda()
        # forward
        out = self.network(fake_data)
        print(out)
        reprod_logger.add("out", out.cpu().detach().numpy())
        reprod_logger.save("../diff/forward_pytorch.npy")

    def metric_pytorch(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        reprod_logger = ReprodLogger()
        self.network.load_state_dict(torch.load("./R2U-Net.pth"))
        self.network.eval()

        with torch.no_grad():
            img = np.load("../fake_img.npy")
            img = Image.fromarray(img)
            target = np.load("../fake_target.npy")
            target = Image.fromarray(target)
            img = transforms.CenterCrop(560)(img)
            img = transforms.ToTensor()(img).unsqueeze(0)
            target = transforms.CenterCrop(560)(target)
            target = np.array(target)
            target_ = target.flatten() / 255
            target_ = target_.astype(np.uint8)


            predict = self.network(img)
            predict = np.squeeze(predict.numpy(), axis=(0, 1))
            predict_ = predict.flatten()
            predict_ = (predict_ >= 0.5).astype(np.uint8)

            TP = np.sum(np.logical_and(predict_ == 1, target_ == 1))  # true positive
            TN = np.sum(np.logical_and(predict_ == 0, target_ == 0))  # true negative
            FP = np.sum(np.logical_and(predict_ == 1, target_ == 0))  # false positive
            FN = np.sum(np.logical_and(predict_ == 0, target_ == 1))  # false negative

            AC = (TP + TN) / (TP + TN + FP + FN)  # accuracy
            SE = (TP) / (TP + FN)  # sensitivity
            SP = TN / (TN + FP)  # specificity
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F1 = 2 * ((precision * recall) / (precision + recall))
            fpr, tpr, _ = roc_curve(target_, predict_)
            AUC = auc(fpr, tpr)
            print('F1 score:', F1)
            reprod_logger.add("F1", np.array(F1))
        reprod_logger.save("../diff/metric_pytorch.npy")


    def loss_pytorch(self):
        np.random.seed(0)
        torch.manual_seed(0)
        reprod_logger = ReprodLogger()
        self.network.cuda()
        self.network.load_state_dict(torch.load("./R2U-Net.pth"))
        self.network.eval()
        # read or gen fake data
        fake_data = np.load("../fake_data.npy")
        fake_data = torch.from_numpy(fake_data).cuda()
        # forward
        fake_label = np.load("../fake_label.npy")
        fake_label = torch.from_numpy(fake_label).cuda()
        predict = self.network(fake_data)

        predict = predict.view(predict.size(0), -1)
        fake_label = fake_label.view(fake_label.size(0), -1)

        loss_fn = nn.BCELoss()
        loss = loss_fn(predict, fake_label)

        print("loss:", loss)
        reprod_logger.add("loss", loss.cpu().detach().numpy())
        reprod_logger.save("../diff/loss_pytorch.npy")
    #
    def bp_align_pytorch(self):
        np.random.seed(0)
        torch.manual_seed(0)
        reprod_logger = ReprodLogger()

        self.network.cuda()
        self.network.load_state_dict(torch.load("./R2U-Net.pth"))
        self.network.train()
        # read or gen fake data
        fake_data = np.load("../fake_data.npy")
        fake_data = torch.from_numpy(fake_data).cuda()
        # forward
        fake_label = np.load("../fake_label.npy")
        fake_label = torch.from_numpy(fake_label).cuda()
        loss_list = []

        optimizer = optim.Adam(lr=self.lr, params=self.network.parameters())
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epoch, eta_min=0.00001)

        for idx in range(4):
            predict = self.network(fake_data)
            predict = predict.view(predict.size(0), -1)
            fake_label = fake_label.view(fake_label.size(0), -1)
            loss_fn = nn.BCELoss()
            loss = loss_fn(predict, fake_label)
            loss.backward()
            print("loss:", loss)
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            print("lr:", scheduler.get_lr())
            loss_list.append([loss.cpu().detach().numpy()])
        print(loss_list)
        reprod_logger.add("loss_list[0]", np.array(loss_list[0]))
        reprod_logger.add("loss_list[1]", np.array(loss_list[1]))
        reprod_logger.add("loss_list[2]", np.array(loss_list[2]))
        reprod_logger.save("../diff/bp_align_pytorch.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='U-Net')
    # general setting
    parser.add_argument('--model', type=str, default='R2U-Net', help='U-Net R2U-Net IterNet')
    parser.add_argument('--mode', type=str, default='train', help='train test')
    parser.add_argument('--dataset_path', type=str, default='./DRIVE/', help='dataset path')
    parser.add_argument('--result_path', type=str, default='./', help='path to save output')
    # training setting
    parser.add_argument('--epoch', type=int, default=45, help='training epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    # testing setting
    parser.add_argument('--show', type=str, default='False', help='if show the predicted image')
    args = parser.parse_args()

    m = model(args)

    #######################################
    # m.save_model()
    # m.show_pkl()
    # m.pytorch2paddle()
    # m.forward_pytorch()
    # m.loss_pytorch()
    m.metric_pytorch()
    # m.bp_align_pytorch()
    #######################################




    # if args.mode == 'train':
    #     m.train()
    # else:
    #     if args.show == 'True':
    #         m.test(True)
    #     else:
    #         m.test(False)