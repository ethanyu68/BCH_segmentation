import os
from collections import OrderedDict
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
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
import json
#from kornia.losses import total_variation

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)

parser = argparse.ArgumentParser()
parser.add_argument("--block_config", default=((4,4,4),(4,4,4)), type=tuple)
parser.add_argument("--batchsize", default=8, type=int)
parser.add_argument("--batchsize_val", default=32, type=int)
parser.add_argument("--imagesize", default=512, type=int)
parser.add_argument("--initial_lr", default=0.0001, type=float, help='initial learning rate')
parser.add_argument("--ep_start", default=1, type=int)
parser.add_argument("--num_classes", default=3, type=int)
parser.add_argument("--lr_reduction", default=0.2, type=float, help='the learning rate will be reduced to <lr_reduction> of current rate at every <step size>')
parser.add_argument("--lr_steps", default=[0, 200, 400, 601], type=int, help='learning rate will be reduced at every <step_size> epoch')
parser.add_argument("--model_ID",default='dense_dmap_0316', help='the model ID to be saved')

parser.add_argument("--path_data",default='./data/npy/etv167_NIFTI_pre_reviewed_0219/', help='the training dataset csv')

parser.add_argument("--path_model",default='./MRI_segmentation_pipeline/models_seg_old.py', help='the model ID to be saved')
parser.add_argument("--path_main",default='./MRI_segmentation_pipeline/train_2025Mar_CV.py', help='the path of main file to be saved')
parser.add_argument("--path_dataset",default='./MRI_segmentation_pipeline/dataset_2025Mar.py', help='the path of datset file to be saved')
parser.add_argument("--ssl", default=False)
parser.add_argument("--distance_map", default=1)
parser.add_argument("--reg_TV", default=1, type=int)
parser.add_argument("--syn_air", default=0, type=int)

parser.add_argument("--path_resume",default="checkpoints_2024/dense_dmap_EAF_0729/model_epoch_200.pth", help='the model ID to be saved')
parser.add_argument("--path_TL",default="checkpoints_2024/dense_dmap_ESTHIBCH_0719/model_epoch_300a.pth", help='the model ID to be saved')
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
    dice_loss = DiceLoss()

    path_ckp = os.path.join('./checkpoints_2025/', opt.model_ID)
    opt.path_ckp = path_ckp
    if not os.path.exists(os.path.join(path_ckp, opt.model_ID)):
        os.makedirs(os.path.join(path_ckp))
        os.makedirs(os.path.join(path_ckp, 'checkpoints'))
    shutil.copy(opt.path_model, path_ckp)
    shutil.copy(opt.path_main, path_ckp)
    shutil.copy(opt.path_dataset, path_ckp)
    # save opt
    with open(path_ckp + '/opt.txt', 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    
    
    # freeze encoder
    # for param in model.encoder.parameters():
    #     param.requires_grad = False
   
    #model = model.cuda()
    for fold in range(1, 6):
        # setting model
        model = DenseTV(num_classes=opt.num_classes, pretrain=False, block_config=opt.block_config)
        #model = UNet(out_channels=4)
        model.apply(init_weights)
        # load the self-supervised trained encoder
        if os.path.exists(opt.path_TL):
            state = torch.load(opt.path_TL)['model']
            encoder = OrderedDict()
            for layer in state.keys():
                if 'encoder' in layer:
                    encoder[layer[15:]] = state[layer]
            model.encoder.load_state_dict(encoder)
            print("===> load the pre-trained encoder:{}".format(opt.path_TL))
            model = torch.nn.DataParallel(model).cuda()
        elif os.path.exists(opt.path_resume):
            model = torch.nn.DataParallel(model).cuda()
            state = torch.load(opt.path_resume)['model']
            model.load_state_dict(state)
            #opt.ep_start = torch.load(opt.path_resume)['epoch']
            print("===> resume the checkpoint:{}".format(opt.path_resume))
        else:
            model = torch.nn.DataParallel(model).cuda()
        opt.path_ckp = os.path.join(opt.path_ckp, 'fold{}'.format(fold))
        if not os.path.exists(opt.path_ckp):
            os.makedirs(opt.path_ckp)
        print("===> Setting Optimizer")
        optimizer = optim.Adam(model.parameters(), lr=opt.initial_lr, weight_decay=1e-6)

        train_set = Dataset(path_fdr=opt.path_data, fold=fold, opt=opt, phase='train', patch=False)
        val_set = Dataset(path_fdr=opt.path_data, fold=fold, opt=opt, phase='val', patch=False)

        train_data_loader = DataLoader(dataset=train_set, num_workers = 1, batch_size=opt.batchsize, shuffle=True)
        val_data_loader = DataLoader(dataset=val_set, batch_size=opt.batchsize_val, shuffle=False)

        results = {'epoch':[0], 'average': [1000], 'background': [1000], 'CSF':[1000], 'tissue':[1000],'air':[1000]}
        for epoch in range(opt.ep_start, opt.lr_steps[-1]):
            train(train_data_loader, optimizer, model, epoch, dice_loss, opt)
        
            avg_loss, per_channel_loss = validate(val_data_loader, model, epoch, dice_loss, opt)
            print("===> validation:{}".format(per_channel_loss))

            if avg_loss < results['average'][-1]:
                save_checkpoint(model, optimizer, epoch, per_channel_loss, filepath=os.path.join(opt.path_ckp, "ckp_{}.pth".format(epoch)))
                results['epoch'].append(epoch)
                results['average'].append(avg_loss)
                results['background'].append(1 -per_channel_loss[0])
                results['CSF'].append(1 - per_channel_loss[1])
                results['tissue'].append(1 -per_channel_loss[2])
                if opt.num_classes == 4:
                    results['air'].append(1 -per_channel_loss[3])
                else:
                    results['air'].append(-1)


            #     df = pd.DataFrame(results)
            #     df.to_csv(path_ckp+'/results.csv')
            
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

def unnormalize(data):
    mean = torch.tensor([0.485, 0.456, 0.406]).view([1, 3, 1, 1])
    std = torch.tensor([0.229, 0.224, 0.225]).view([1, 3, 1, 1])
    orig = data * std + mean
    return orig


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
