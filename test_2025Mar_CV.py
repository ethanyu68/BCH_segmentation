import os
from collections import OrderedDict
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import cv2
import torch.nn.functional as F
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
#from torchinfo import summary
from torch.utils.data import DataLoader
from dataset_2025Mar import Dataset
import matplotlib.pyplot as plt
from models_seg_old import DenseTV, DenseSETV, UNet
#import torchgeometry as tgm
import torchvision.models as models
import numpy as np
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--block_config", default=((4,4,4),(4,4,4)), type=tuple)
parser.add_argument("--batchsize", default=8, type=int)
parser.add_argument("--batchsize_val", default=16, type=int)
parser.add_argument("--imagesize", default=448, type=int)

parser.add_argument("--num_classes", default=3, type=int)
parser.add_argument("--model_ID",default='dense_dmap_etv167preop', help='the model ID to be saved')

parser.add_argument("--path_data",default='./data/npy/etv167_NIFTI_pre_reviewed_0219/', help='the training dataset csv')

parser.add_argument("--distance_map", default=1)
parser.add_argument("--reg_TV", default=1, type=int)
parser.add_argument("--syn_air", default=0, type=int)

parser.add_argument("--path_resume",default="./checkpoints_2025/dense_dmap_etv167preop/fold1/ckp_429.pth", help='the model ID to be saved')
##
parser.add_argument("--notation", default='Description:.')
opt= parser.parse_args()
def main():
    np.random.seed(1111)
    print(opt)
    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found")
    #os.environ['CUDA_VISIBLE_DEVICES']='0'
    #os.environ['NCCL_SHM_DISABLE']='1'
    #os.environ["NCCL_DEBUG"] = "INFO"
    opt.expm = os.path.join(opt.path_resume.split('/')[0], opt.path_resume.split('/')[1], opt.path_resume.split('/')[2])
    dice_loss = DiceLoss()
    # setting model
    fold = 1
    model = DenseTV(num_classes=opt.num_classes, pretrain=False, block_config=opt.block_config)
    state = torch.load(opt.path_resume)['model_state_dict']
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(state)
    print("===> resume the checkpoint:{}".format(opt.path_resume))

    test_set = Dataset(path_fdr=opt.path_data, fold=fold, opt=opt, phase='test', patch=False)
    test_data_loader = DataLoader(dataset=test_set, batch_size=opt.batchsize_val, shuffle=False)
    
    avg_loss, per_channel_loss = test(test_data_loader, model, dice_loss, opt)
    print("===> Test fold {}:{}".format(fold, per_channel_loss))
            
def train(training_data_loader, optimizer, model, epoch, dice_loss, opt):
    step = [i for i, step in enumerate(opt.lr_steps) if epoch > step][-1]
    lr = opt.initial_lr * (opt.lr_reduction ** step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    csf_loss, brain_loss = 0, 0
    for iteration, batch in enumerate(training_data_loader, 1):
        data, label_all, _, = batch
        out, out3, out4, out5 = model(data.to('cuda'))
        pred = out.softmax(dim=1)
        label = one_hot_encoding(label_all, num_classes=opt.num_classes).cuda()
        #out = model(data.cuda())
        # no eaf
       
        loss_dice, dice_c = dice_loss(pred, label)
        # num_eaf = torch.sum(eaf)
        # if num_eaf<data.shape[0]:
        #     _, per_channel_loss = dice_loss(out[eaf.ravel() == 0], expand_as_one_hot(label_all[:, 0, :, :][eaf.ravel() == 0].long().cuda(),
        #                                                                 C=opt.num_classes))  # out[1] = 1, if there is objec
        #     dice += torch.mean(per_channel_loss[0] + per_channel_loss[2] + per_channel_loss[4]) # background, brain, air
        # # eaf
        # if num_eaf>0:
        #     dice2, per_channel_loss = dice_loss(out[eaf.ravel() == 1], expand_as_one_hot(label_all[:, 0, :, :][eaf.ravel() == 1].long().cuda(),
        #                                                                 C=opt.num_classes))  # out[1] = 1, if there is objec
        #     dice += dice2 * 2
        
        if opt.reg_TV == 1:
            TV = getTV(out4)
        else:
            TV = 0
        loss = loss_dice + 0.01 * TV
        #csf_loss += per_channel_loss[1].cpu().detach().numpy()
        #brain_loss += per_channel_loss[2].cpu().detach().numpy()
        if iteration % 10 == 0:
            print("===> Epoch[{}]({}), lr: {}, dice:{}, TV:{:.6f}".format(epoch, iteration, lr, np.mean(dice_c.cpu().detach().numpy(), 0), TV))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # csf_loss /= iteration
    # brain_loss /= iteration 
    # if os.path.exists(opt.path_ckp + '/training_loss.csv'):
    #     df = pd.read_csv(opt.path_ckp + '/training_loss.csv')
    #     df.append(pd.DataFrame({'epoch':[epoch], 'csf_loss':[csf_loss], 'brain_loss':[brain_loss]}))
    # else:
    #     df = pd.DataFrame({'epoch':[epoch], 'csf_loss':[csf_loss], 'brain_loss':[brain_loss]})
    # df.to_csv(opt.path_ckp + '/training_loss.csv')

def getTV_mask(feat_map, region_mask):
    res1 = torch.abs(feat_map[:, :, 1:,:] - feat_map[:, :, :-1, :])[:, :, :, :-1] 
    res2 = torch.abs(feat_map[:, :, :, 1:] - feat_map[:, :, :, :-1])[:, :, 1:, :]
    TVCSF = torch.mean(res1 *region_mask[:, :, 1:,1:].cuda()) + torch.mean(res2 * region_mask[:, :, 1:, 1:].cuda())
    res1, res2 = None, None
    return TVCSF

def getTV(feat_map):
    res1 = torch.abs(feat_map[:, :, 1:,:] - feat_map[:, :, :-1, :])[:, :, :, :-1] 
    res2 = torch.abs(feat_map[:, :, :, 1:] - feat_map[:, :, :, :-1])[:, :, 1:, :]
    TVCSF = torch.mean(res1) + torch.mean(res2)
    return TVCSF

def validate(val_data_loader, model, epoch, criterion, opt):
    model.eval()
    per_channel = []
    for iteration, batch in enumerate(val_data_loader, 1):
        with torch.no_grad():
            data, label,  _  = batch
            out, _, _, _ = model(data.cuda())
            #out = model(data.cuda())
            loss_all, per_channel_loss = criterion(out.softmax(dim=1), one_hot_encoding(label.cuda(), opt.num_classes))
            per_channel.append(per_channel_loss)

    per_channel = torch.mean(torch.cat(per_channel, 0),0).cpu().numpy()
    return 1 - np.mean(per_channel[:2]), per_channel

def test(val_data_loader, model,criterion, opt):
    model.eval()
    per_channel = []
    for iteration, batch in enumerate(val_data_loader, 1):
        with torch.no_grad():
            data, label,  fname  = batch
            out, _, _, _ = model(data.cuda())
            #out = model(data.cuda())
            loss_all, per_channel_loss = criterion(out[:,:,::2, ::2].softmax(dim=1), one_hot_encoding(label[:,::2, ::2].cuda(), opt.num_classes))
            per_channel.append(per_channel_loss)
            #save_fig(data, out, label, fname, opt.expm)

    per_channel = torch.mean(torch.cat(per_channel, 0),0).cpu().numpy()
    return 1 - np.mean(per_channel[:2]), per_channel

def unnormalize(data):
    mean = torch.tensor([0.485, 0.456, 0.406]).view([1, 3, 1, 1])
    std = torch.tensor([0.229, 0.224, 0.225]).view([1, 3, 1, 1])
    orig = data * std + mean
    return orig
def save_fig(input, output, label, filenames,save_path):
    if not os.path.exists(opt.expm + '/results'):
        os.makedirs(opt.expm + '/results')
    mri = unnormalize(input).cpu().numpy()
    output = torch.argmax(output, 1).cpu().numpy()
    label = label.cpu().numpy()
    
    for i in range(input.shape[0]):
        combo=np.concatenate([mri[i][0] * 255, output[i]/2*255 , label[i]/2 *255], 1)
        path = opt.expm + '/results/' + filenames[0].split('/')[-1] + '_' +str(i) + '.jpg'
        cv2.imwrite(path, combo)
        # plt.figure()
        # plt.subplot(1,3,1)
        # plt.imshow(mri[i][0])
        # plt.subplot(1,3,2)
        # plt.imshow(output[i])
        # plt.subplot(1,3,3)
        # plt.imshow(label[i])
        # plt.savefig(opt.expm + '/results/' + filenames[0].split('/')[-1] + '_' +str(i) + '.jpg')
        # plt.close()
        


def one_hot_encoding(target, num_classes):
    """
    Converts a target segmentation map to one-hot encoding.
    
    Args:
        target (Tensor): Shape (B, H, W) with class indices
        num_classes (int): Number of segmentation classes

    Returns:
        one_hot (Tensor): Shape (B, C, H, W) with one-hot encoded labels
    """
    # Ensure target has shape (B, H, W)
    assert len(target.shape) == 3, "Target must have shape (B, H, W)"

    # Apply one-hot encoding
    one_hot = F.one_hot(target.long(), num_classes)  # Shape: (B, H, W, C)
    one_hot = one_hot.permute(0, 3, 1, 2)  # Reshape to (B, C, H, W)

    return one_hot.float()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Compute Dice Loss for multi-class segmentation.
        
        Args:
        - pred (Tensor): Predicted output of shape (B, C, H, W) or (B, C, D, H, W)
        - target (Tensor): Ground truth of same shape (one-hot encoded)

        Returns:
        - Dice Loss (Tensor): Scalar value
        """
        assert pred.shape == target.shape, "Pred and target must have the same shape"

        # Flatten all except the batch and channel dimensions
        pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        target = target.contiguous().view(target.shape[0], target.shape[1], -1)

        # Compute intersection and union
        intersection = (pred * target).sum(dim=2)
        denominator = pred.sum(dim=2) + target.sum(dim=2)

        # Compute Dice score per class and take the mean over channels
        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1 - dice.mean()

        return dice_loss, dice

def save_checkpoint(model, optimizer, epoch, loss, filepath="checkpoint.pth"):
    """
    Saves the model checkpoint.
    
    Args:
        model (torch.nn.Module): The trained model.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch number.
        loss (float): Training or validation loss.
        filepath (str): Path to save the checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at {filepath} (Epoch {epoch})")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
