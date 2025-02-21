import os
from collections import OrderedDict
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
#from torchinfo import summary
from torch.utils.data import DataLoader
from dataset_2024Aug import Dataset
import matplotlib.pyplot as plt
from models_seg_old import DenseTV, DenseSETV, UNet
#import torchgeometry as tgm
import torchvision.models as models
import numpy as np
from losses import DiceLoss, expand_as_one_hot
import argparse
from utils import *
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
parser.add_argument("--batchsize", default=16, type=int)
parser.add_argument("--batchsize_val", default=32, type=int)
parser.add_argument("--imagesize", default=512, type=int)
parser.add_argument("--initial_lr", default=0.0001, type=float, help='initial learning rate')
parser.add_argument("--ep_start", default=100, type=int)
parser.add_argument("--num_classes", default=5, type=int)
parser.add_argument("--lr_reduction", default=0.2, type=float, help='the learning rate will be reduced to <lr_reduction> of current rate at every <step size>')
parser.add_argument("--lr_steps", default=[0, 200, 400, 601], type=int, help='learning rate will be reduced at every <step_size> epoch')
parser.add_argument("--model_ID",default='dense_dmap_EAF_0804', help='the model ID to be saved')

parser.add_argument("--path_tr_ls",default='data/data_for_training/trainingdata_esthiBCH4cls_BCH5cls_0804.csv', help='the training dataset csv')

parser.add_argument("--path_model",default='MRI_segmentation_pipeline/models_seg_old.py', help='the model ID to be saved')
parser.add_argument("--path_main",default='MRI_segmentation_pipeline/train_2024Aug_EAF.py', help='the path of main file to be saved')
parser.add_argument("--path_dataset",default='MRI_segmentation_pipeline/dataset_2024Aug.py', help='the path of datset file to be saved')
parser.add_argument("--ssl", default=False)
parser.add_argument("--distance_map", default=1)
parser.add_argument("--reg_TV", default=1, type=int)
parser.add_argument("--syn_air", default=1, type=int)

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
    dice_loss = DiceLoss(sigmoid_normalization=False)

    path_ckp = os.path.join('checkpoints_2024/', opt.model_ID)
    opt.path_ckp = path_ckp
    checkdirctexist(path_ckp)
    shutil.copy(opt.path_model, path_ckp)
    shutil.copy(opt.path_main, path_ckp)
    shutil.copy(opt.path_dataset, path_ckp)
    # save opt
    with open(path_ckp + '/opt.txt', 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

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
    if os.path.exists(opt.path_resume):
        model = torch.nn.DataParallel(model).cuda()
        state = torch.load(opt.path_resume)['model']
        model.load_state_dict(state)
        #opt.ep_start = torch.load(opt.path_resume)['epoch']
        print("===> resume the checkpoint:{}".format(opt.path_resume))
        
    
    
    # freeze encoder
    # for param in model.encoder.parameters():
    #     param.requires_grad = False
   
    #model = model.cuda()

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.initial_lr, weight_decay=1e-6)

    train_set = Dataset(path_csv=opt.path_tr_ls, opt=opt, aug=True, patch=False)
    #val_set = Dataset(path_csv=opt.path_val_ls, opt=opt, aug=False, patch=False)

    train_data_loader = DataLoader(dataset=train_set, num_workers = 1, batch_size=opt.batchsize, shuffle=True)
    #val_data_loader = DataLoader(dataset=val_set, batch_size=opt.batchsize_val, shuffle=False)

    results = {'epoch':[], 'average': [], 'background': [], 'CSF':[], 'tissue':[],'air':[]}
    for epoch in range(opt.ep_start, opt.lr_steps[-1]):
        train(train_data_loader, optimizer, model, epoch, dice_loss, opt)
        # if epoch % 10 == 0:
        #     per_channel_loss = validate(val_data_loader, model, epoch, dice_loss, opt)
        #     print("===> validation:{}".format(per_channel_loss))
        #     results['epoch'].append(epoch)
        #     results['average'].append(1 - np.mean(per_channel_loss[:2]))
        #     results['background'].append(1 -per_channel_loss[0])
        #     results['CSF'].append(1 - per_channel_loss[1])
        #     results['tissue'].append(1 -per_channel_loss[2])
        #     if opt.num_classes == 4:
        #         results['air'].append(1 -per_channel_loss[3])
        #     else:
        #         results['air'].append(-1)


        #     df = pd.DataFrame(results)
        #     df.to_csv(path_ckp+'/results.csv')
        if epoch > 199 and epoch % 50 == 0:
            save_checkpoint(model, epoch, optimizer, opt, path_ckp)

def train(training_data_loader, optimizer, model, epoch, dice_loss, opt):
    step = [i for i, step in enumerate(opt.lr_steps) if epoch > step][-1]
    lr = opt.initial_lr * (opt.lr_reduction ** step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    csf_loss, brain_loss = 0, 0
    for iteration, batch in enumerate(training_data_loader, 1):
        data, label_all, _, eaf = batch
        out, out3, out4, out5 = model(data.to('cuda'))
        #out = model(data.cuda())
        # no eaf
        dice = 0
        num_eaf = torch.sum(eaf)
        if num_eaf<data.shape[0]:
            _, per_channel_loss = dice_loss(out[eaf.ravel() == 0], expand_as_one_hot(label_all[:, 0, :, :][eaf.ravel() == 0].long().cuda(),
                                                                        C=opt.num_classes))  # out[1] = 1, if there is objec
            dice += torch.mean(per_channel_loss[0] + per_channel_loss[2] + per_channel_loss[4]) # background, brain, air
        # eaf
        if num_eaf>0:
            dice2, per_channel_loss = dice_loss(out[eaf.ravel() == 1], expand_as_one_hot(label_all[:, 0, :, :][eaf.ravel() == 1].long().cuda(),
                                                                        C=opt.num_classes))  # out[1] = 1, if there is objec
            dice += dice2 * 2
        
        if opt.reg_TV == 1:
            TV = getTV(out4)
        else:
            TV = 0
        loss = dice + 0.01 * TV
        #csf_loss += per_channel_loss[1].cpu().detach().numpy()
        #brain_loss += per_channel_loss[2].cpu().detach().numpy()
        if iteration % 10 == 0:
            print("===> Epoch[{}]({}), lr: {}, dice:{}, TV:{:.6f}".format(epoch, iteration, lr, per_channel_loss.cpu().detach().numpy(), TV))

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
    per_channel = torch.tensor([0,0,0,0]).float().cuda()
    for iteration, batch in enumerate(val_data_loader, 1):
        with torch.no_grad():
            data, label,  _  = batch
            out, _, _, _ = model(data.cuda())
            #out = model(data.cuda())
            loss_all, per_channel_loss = criterion(out, expand_as_one_hot(label[:, 0, :, :].long().cuda(), C=opt.num_classes))
            per_channel += per_channel_loss

    per_channel = per_channel/iteration
    return per_channel.cpu().numpy()



def unnormalize(data):
    mean = torch.tensor([0.485, 0.456, 0.406]).view([1, 3, 1, 1])
    std = torch.tensor([0.229, 0.224, 0.225]).view([1, 3, 1, 1])
    orig = data * std + mean
    return orig


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
