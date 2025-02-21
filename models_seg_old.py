import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseSELayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dilation=1, memory_efficient=False):
        super(_DenseSELayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('se', _SE(bn_size * growth_rate, bn_size * growth_rate//8))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=dilation,dilation=dilation,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        bottleneck_output = bn_function(*prev_features)
        channel_attention_output = self.se(bottleneck_output)
        new_features = self.conv2(self.relu2(self.norm2(channel_attention_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features



class _DenseSEBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, num_output_features, dilation = 1, bn_size=4, growth_rate=32, drop_rate=0, memory_efficient=False):
        super(_DenseSEBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseSELayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                dilation=dilation,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)
        #self.add_module('se', _SE(num_input_features + num_layers* growth_rate, num_input_features + num_layers* growth_rate//16))
        self.add_module('trans', nn.Conv2d(num_input_features + num_layers* growth_rate, num_output_features, kernel_size=(1,1), stride=(1,1)))
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            if name != 'trans':
                new_features = layer(*features)
                features.append(new_features)
        #se = self.se(torch.cat(features, 1))
        return self.trans(torch.cat(features, 1))




class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dilation=1, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=dilation,dilation=dilation,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features



class _DenseCBAMLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseCBAMLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('cbam', CBAM(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),

        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.cbam(self.norm2(bottleneck_output))))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features



class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, num_output_features, dilation = 1, bn_size=4, growth_rate=32, drop_rate=0, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                dilation=dilation,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)
        #self.add_module('se', _SE(num_input_features + num_layers* growth_rate, num_input_features + num_layers* growth_rate//16))
        self.add_module('trans', nn.Conv2d(num_input_features + num_layers* growth_rate, num_output_features, kernel_size=(1,1), stride=(1,1)))
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            if name != 'trans':
                new_features = layer(*features)
                features.append(new_features)
        #se = self.se(torch.cat(features, 1))
        return self.trans(torch.cat(features, 1))



class _DenseCBAMBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, num_output_features, bn_size=4, growth_rate=32, drop_rate=0, memory_efficient=False):
        super(_DenseCBAMBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseCBAMLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)
        #self.add_module('se', _SE(num_input_features + num_layers* growth_rate, num_input_features + num_layers* growth_rate//16))
        self.add_module('trans', nn.Conv2d(num_input_features + num_layers* growth_rate, num_output_features, kernel_size=(1,1), stride=(1,1)))
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            if name != 'trans':
                new_features = layer(*features)
                features.append(new_features)
        #se = self.se(torch.cat(features, 1))
        return self.trans(torch.cat(features, 1))




class TransitionBlockEncoder(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlockEncoder, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=(1,1), stride=(1,1),
                                        padding=(0,0), bias=False)
        self.pool1 = nn.MaxPool2d(2,stride=2)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return self.pool1(out)


class TransitionBlockDecoder(nn.Module):
    def __init__(self, in_planes, out_planes, cubic = False, dropRate=0.0):
        super(TransitionBlockDecoder, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.droprate = dropRate
        self.cubic = cubic

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        if self.cubic:
            return F.upsample_bilinear(out, scale_factor=2)
        else:
            return F.upsample_nearest(out, scale_factor=2)


class Encoder(nn.Module):
    def __init__(self, pretrain = True, inter_planes=128, block_config=(4, 4,4), growth_rate =32):
        super(Encoder, self).__init__()
        ############# Encoder 0 - 256 ##############
        self.conv0 = models.densenet121(pretrained = pretrain).features.conv0
        self.pool0 = models.densenet121().features.pool0
        self.dense1 = _DenseBlock(num_layers=block_config[0], num_input_features=64, num_output_features=inter_planes[0],dilation=1,
                                  growth_rate=growth_rate)
        self.trans1 = TransitionBlockEncoder(inter_planes[0], inter_planes[1])
        self.dense2 = _DenseBlock(num_layers=block_config[1], num_input_features=inter_planes[1], num_output_features=inter_planes[1],dilation=2,
                                  growth_rate=growth_rate)
        self.trans2 = TransitionBlockEncoder(inter_planes[1], inter_planes[2])
        ############# Encoder 3 - 32 ##########################
        self.dense3 = _DenseBlock(num_layers=block_config[2], num_input_features=inter_planes[2], num_output_features=inter_planes[2],dilation=2,
                                  growth_rate=growth_rate)
        #self.trans3 = TransitionBlockEncoder(inter_planes[2], inter_planes[3])
        # self.dense4 = _DenseBlock(num_layers=block_config[2], num_input_features=inter_planes[3], num_output_features=inter_planes[3],
        #                           growth_rate=growth_rate)
        ############# Decoder 0 -32 ##############################
    def forward(self, x):
        out0 = self.dense1(self.pool0(self.conv0(x)))
        out1 = self.dense2(self.trans1(out0))
        out2 = self.dense3(self.trans2(out1))
        return out0, out1, out2


class EncoderSE(nn.Module):
    def __init__(self, pretrain = True, inter_planes=128, block_config=(4, 4,4), growth_rate =32):
        super(EncoderSE, self).__init__()
        ############# Encoder 0 - 256 ##############
        self.conv0 = models.densenet121(pretrained = pretrain).features.conv0
        self.pool0 = models.densenet121().features.pool0
        self.dense1 = _DenseBlock(num_layers=block_config[0], num_input_features=64, num_output_features=inter_planes[0],
                                  growth_rate=growth_rate)
        self.trans1 = TransitionBlockEncoder(inter_planes[0], inter_planes[1])
        self.dense2 = _DenseBlock(num_layers=block_config[1], num_input_features=inter_planes[1], num_output_features=inter_planes[1],dilation=1,
                                  growth_rate=growth_rate)
        self.trans2 = TransitionBlockEncoder(inter_planes[1], inter_planes[2])
        ############# Encoder 3 - 32 ##########################
        self.dense3 = _DenseSEBlock(num_layers=block_config[2], num_input_features=inter_planes[2], num_output_features=inter_planes[2],
                                  growth_rate=growth_rate)
        #self.trans3 = TransitionBlockEncoder(inter_planes[2], inter_planes[3])
        # self.dense4 = _DenseBlock(num_layers=block_config[2], num_input_features=inter_planes[3], num_output_features=inter_planes[3],
        #                           growth_rate=growth_rate)
        ############# Decoder 0 -32 ##############################
    def forward(self, x):
        out0 = self.dense1(self.pool0(self.conv0(x)))
        out1 = self.dense2(self.trans1(out0))
        out2 = self.dense3(self.trans2(out1))
        return out0, out1, out2


class Encoder_5input_2branch(nn.Module):
    def __init__(self, pretrain = True, inter_planes=256, out_planes=256, block_config=(6, 4, 4), growth_rate =32):
        super(Encoder_5input_2branch, self).__init__()
        ############# Encoder 0 - 256 ##############
        self.dense1 = nn.Sequential(
            models.densenet121(pretrained=pretrain).features.conv0,
            models.densenet121().features.pool0,
            models.densenet121(pretrained=pretrain).features.denseblock1
        )
        self.conv3D_0 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3,3,3), padding=1, stride = (1,1,1)),
            #nn.BatchNorm3d(num_features = 256),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=256, out_channels=inter_planes, kernel_size=(3, 3, 3), padding=(0, 1, 1),
                      stride=(1, 1, 1)),
            #nn.BatchNorm3d(num_features=inter_planes),
            nn.LeakyReLU(),
        )
        self.conv3D_1 = nn.Sequential(
            nn.Conv3d(in_channels=inter_planes, out_channels=inter_planes, kernel_size=(3, 3, 3), padding=1, stride=(1, 1, 1)),
            #nn.BatchNorm3d(num_features=256),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=256, out_channels=inter_planes, kernel_size=(3, 3, 3), padding=(0, 1, 1),
                      stride=(1, 1, 1)),
            #nn.BatchNorm3d(num_features=inter_planes),
            nn.LeakyReLU(),
        )
        self.trans1_axil = nn.Sequential(TransitionBlockEncoder(256, inter_planes))
        self.trans1_main = TransitionBlockEncoder(256, inter_planes)
        self.dense2_axil = _DenseCBAMBlock(num_layers=4, num_input_features=inter_planes,
                                  num_output_features=inter_planes,
                                  growth_rate=growth_rate)

        self.att_trans1 = _Res4AttentionBlock(num_input_features=inter_planes)
        self.dense2_main = _DenseCBAMBlock(num_layers=block_config[0], num_input_features=inter_planes,
                                           num_output_features=inter_planes,
                                           growth_rate=growth_rate)
        self.trans2_axil = nn.Sequential(TransitionBlockEncoder(inter_planes, inter_planes))
        self.trans2_main = TransitionBlockEncoder(inter_planes, inter_planes)
        self.dense3_main = _DenseCBAMBlock(num_layers=block_config[1], num_input_features=inter_planes,
                                           num_output_features=out_planes,
                                        growth_rate=growth_rate)
        #self.dense3_axil = _DenseCBAMBlock(num_layers=4, num_input_features=inter_planes,
        #                                   num_output_features=inter_planes,
        #                                   growth_rate=growth_rate)
        self.trans3_main = TransitionBlockEncoder(inter_planes, inter_planes)
        self.att_trans2 = _ResAttentionBlock(num_input_features=inter_planes)

        ############# Decoder 0 -32 ##############################
    def forward(self, x):
        #out_dense1_0 = self.dense1(x[:, 0, :, :, :])
        out_dense1_1 = self.dense1(x[:, 1, :, :, :])
        out_dense1_2 = self.dense1(x[:, 2, :, :, :])
        out_dense1_3 = self.dense1(x[:, 3, :, :, :])
        #out_dense1_4 = self.dense1(x[:, 4, :, :, :]) # B x 256 x 128 x 128

        #input012 = torch.cat([out_dense1_0[:, :, None, :, :], out_dense1_1[:, :, None, :, :], out_dense1_2[:, :, None, :, :]], 2)
        input123 = torch.cat([out_dense1_1[:, :, None, :, :], out_dense1_2[:, :, None, :, :], out_dense1_3[:, :, None, :, :]], 2)
        #input234 = torch.cat([out_dense1_2[:, :, None, :, :], out_dense1_3[:, :, None, :, :], out_dense1_4[:, :, None, :, :]], 2) # B x 768 x 128 x 128

        #out_dense1_012 = self.dense1(x[:, 0:3, 0, :, :])
        #out_dense1_123 = self.dense1(x[:, 1:4, 0, :, :])
        #out_dense1_234 = self.dense1(x[:, 2:5, 0, :, :])
        #out_conv3d_axil012 = self.conv3D_0(input012)[:, :, 0, :, :]
        out_conv3d_axil123 = self.conv3D_0(input123)[:, :, 0, :, :]
        #out_conv3d_axil234 = self.conv3D_0(input234)[:, :, 0, :, :]

        #out_trans1_axil012 = self.trans1_axil(out_conv3d_axil012)
        out_trans1_axil123 = self.trans1_axil(out_conv3d_axil123)
        #out_trans1_axil234 = self.trans1_axil(out_conv3d_axil234)
        out_trans1_main = self.trans1_main(out_dense1_2)

        #out_dense2_axil012 = self.dense2_axil(out_trans1_axil012)
        out_dense2_axil123 = self.dense2_axil(out_trans1_axil123)
        #out_dense2_axil234 = self.dense2_axil(out_trans1_axil234)
        out_dense2_main = self.dense2_main(out_trans1_main)
        #out_dense2 = torch.cat([out_dense2_axil012[:, :, None, :, :], out_dense2_axil123[:, :, None, :, :], out_dense2_axil234[:, :, None, :, :]], 2)
        #out_conv3d_2 = self.conv3D_1(out_dense2)[:, :, 0, :, :]

        #out_dense2_axil = self.dense2_axil(out_conv3d_2)
        out_trans2_axil = self.trans2_axil(out_dense2_axil123) # B x 256 x 32 x 32
        out_trans2_main = self.trans2_main(out_dense2_main)
        out_att = self.att_trans2(out_trans2_main, out_trans2_axil)
        out_dense3_axil = self.dense3_main(out_att)
        #out_trans1_main = self.trans1_main(out_dense1_2) # # B x 256 x 64 x 64

        # out_att_trans1 = self.att_trans1(out_trans1_main, out_dense2_axil012,
        #                                    out_dense2_axil123, out_dense2_axil234) # B x 256 x 64 x 64
        #out_dense2_main = self.dense2_main(out_att_trans1) # B x 256 x 64 x 64
        #out_trans2_main = self.trans2_main(out_dense2_main) # B x 256 x 32 x 32

        #out_att_trans2 = self.att_trans2(out_trans2_main, out_trans2_axil)
        #out_dense3_main = self.dense3_main(out_att_trans2) # B x 256 x 32 x 32
        return out_dense1_2, out_dense2_axil123, out_dense3_axil, 0



class Encoder_5input_2branch_2(nn.Module):
    def __init__(self, pretrain = True, inter_planes=128, out_planes=256, block_config=(6, 4, 4), growth_rate =32):
        super(Encoder_5input_2branch_2, self).__init__()
        ############# Encoder 0 - 256 ##############
        self.conv0 = models.densenet121(pretrained=pretrain).features.conv0
        self.conv0_axil = nn.Conv2d(5, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)
        self.pool0 = models.densenet121().features.pool0
        # dense 1
        self.dense1 = models.densenet121(pretrained=pretrain).features.denseblock1
        self.dense1_axil = models.densenet121(pretrained=pretrain).features.denseblock1
        # 128 x 128
        # tran 1
        self.trans1_axil = TransitionBlockEncoder(256, inter_planes)
        self.trans1_main = TransitionBlockEncoder(256, inter_planes)
        # 64 x 64
        # att 1
        self.att_trans1 = _ResAttentionBlock(num_input_features=inter_planes)
        # dense 2
        num_feat = inter_planes
        self.dense2 = _DenseCBAMBlock(num_layers=block_config[0], num_input_features=num_feat,
                                      num_output_features=(block_config[0]*growth_rate + num_feat)//2,
                                      growth_rate=growth_rate)
        self.dense2_axil = _DenseCBAMBlock(num_layers=block_config[0], num_input_features=num_feat,
                                           num_output_features=(block_config[0]*growth_rate + num_feat)//2,
                                           growth_rate=growth_rate)
        num_feat = (block_config[0]*growth_rate + num_feat)//2
        # tran 2
        self.trans2_axil = TransitionBlockEncoder(num_feat, num_feat)
        self.trans2_main = TransitionBlockEncoder(num_feat, num_feat)
        #
        self.att_trans2 = _ResAttentionBlock(num_input_features=num_feat)
        # dense 3
        self.dense3 = _DenseCBAMBlock(num_layers=block_config[1], num_input_features=num_feat,
                                      num_output_features=(block_config[1] * growth_rate + num_feat) // 2,
                                      growth_rate=growth_rate)
        # self.dense3_axil = _DenseCBAMBlock(num_layers=block_config[1], num_input_features=num_feat,
        #                                    num_output_features=(block_config[1] * growth_rate + num_feat) // 2,
        #                                    growth_rate=growth_rate)
        ############# Decoder 0 -32 ##############################
    def forward(self, x):
        out_conv0 = self.pool0(self.conv0(x[:, 2, :, :, :]))
        out_conv0_axil = self.pool0(self.conv0_axil(x[:, :, 0, :, :]))

        out_dense1 = self.dense1(out_conv0)
        out_dense1_axil = self.dense1_axil(out_conv0_axil)

        out_trans1 = self.trans1_main(out_dense1)
        out_trans1_axil = self.trans1_axil(out_dense1_axil)

        out_att1 = self.att_trans1(out_trans1, out_trans1_axil)

        out_dense2 = self.dense2(out_att1)
        out_dense2_axil = self.dense2_axil(out_trans1_axil)

        out_trans2 = self.trans2_main(out_dense2)
        out_trans2_axil = self.trans2_axil(out_dense2_axil)

        out_att2 = self.att_trans2(out_trans2, out_trans2_axil)

        out_dense3 = self.dense3(out_att2)
        #out_dense3_axil = self.dense3_axil(out_trans2_axil)


        return out_dense1, out_dense2, out_dense3, out_trans2_axil



class Encoder_5input_2branch_3(nn.Module):
    def __init__(self, pretrain = True, inter_planes=128, out_planes=256, block_config=(6, 4, 4), growth_rate =32):
        super(Encoder_5input_2branch_3, self).__init__()
        ############# Encoder 0 - 256 ##############
        self.conv0 = models.densenet121(pretrained=pretrain).features.conv0
        self.conv0_axil = nn.Conv2d(5, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)
        self.pool0 = models.densenet121().features.pool0
        # dense 1
        self.dense1 = _DenseCBAMBlock(num_layers=6, num_input_features=64,
                                      num_output_features=(6*growth_rate + 64)//2,
                                      growth_rate=growth_rate)
        self.dense1_axil = models.densenet121(pretrained=pretrain).features.denseblock1
        # 128 x 128
        # tran 1
        self.trans1_axil = TransitionBlockEncoder(128, inter_planes)
        self.trans1_main = TransitionBlockEncoder(256, inter_planes)
        # 64 x 64
        # att 1
        self.att_trans1 = _ResAttentionBlock(num_input_features=inter_planes)
        # dense 2
        num_feat = inter_planes
        self.dense2 = _DenseCBAMBlock(num_layers=block_config[0], num_input_features=num_feat,
                                      num_output_features=(block_config[0]*growth_rate + num_feat)//2,
                                      growth_rate=growth_rate)
        self.dense2_axil = _DenseCBAMBlock(num_layers=block_config[0], num_input_features=num_feat,
                                           num_output_features=(block_config[0]*growth_rate + num_feat)//2,
                                           growth_rate=growth_rate)
        num_feat = (block_config[0]*growth_rate + num_feat)//2
        # tran 2
        self.trans2_axil = TransitionBlockEncoder(num_feat, num_feat)
        self.trans2_main = TransitionBlockEncoder(num_feat*2, num_feat)
        #
        self.att_trans2 = _ResAttentionBlock(num_input_features=num_feat)
        # dense 3
        self.dense3 = _DenseCBAMBlock(num_layers=block_config[1], num_input_features=num_feat,
                                      num_output_features=(block_config[1] * growth_rate + num_feat) // 2,
                                      growth_rate=growth_rate)
        self.dense3_axil = _DenseCBAMBlock(num_layers=block_config[1], num_input_features=num_feat,
                                            num_output_features=(block_config[1] * growth_rate + num_feat) // 2,
                                            growth_rate=growth_rate)
        ############# Decoder 0 -32 ##############################
    def forward(self, x):
        out_conv0 = self.pool0(self.conv0(x[:, 2, :, :, :]))
        out_conv0_axil = self.pool0(self.conv0_axil(x[:, :, 0, :, :]))

        out_dense1 = self.dense1(out_conv0)
        out_dense1_axil = self.dense1_axil(out_conv0_axil)

        out_trans1 = self.trans1_main(out_dense1)
        out_trans1_axil = self.trans1_axil(out_dense1_axil)

        out_dense2 = self.dense2(out_trans1)
        out_dense2_axil = self.dense2_axil(out_trans1_axil)

        out_trans2 = self.trans2_main(torch.cat([out_dense2, out_dense2_axil], 1))
        #out_trans2 = self.trans2_main(out_dense2)
        #out_trans2_axil = self.trans2_axil(out_dense2_axil)

        out_dense3 = self.dense3(out_trans2)
        #out_dense3_axil = self.dense3_axil(out_trans2_axil)


        return out_dense1, out_dense2, out_dense3, out_trans1_axil




class Encoder_5input(nn.Module):
    def __init__(self, pretrain = True, inter_planes=256, out_planes=256, block_config=(6, 4, 4), growth_rate =32):
        super(Encoder_5input, self).__init__()
        ############# Encoder 0 - 256 ##############
        self.conv00 = models.densenet121(pretrained = pretrain).features.conv0
        self.pool00 = models.densenet121().features.pool0
        #
        self.conv_tran = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=(3,3), padding=1),
            nn.ReLU()
        )
        self.dense1 = models.densenet121(pretrained=pretrain).features.denseblock1
        self.trans1 = TransitionBlockEncoder(256, 128)
        #self.dense2 = models.densenet121(pretrained=pretrain).features.denseblock2
        self.dense2 = _DenseBlock(num_layers=block_config[0], num_input_features=512,
                                  num_output_features=inter_planes,
                                  growth_rate=growth_rate)
        self.trans2 = TransitionBlockEncoder(512, inter_planes)
        ############# Encoder 3 - 32 ##########################
        self.dense3 = _DenseCBAMBlock(num_layers=block_config[1], num_input_features=inter_planes, num_output_features=inter_planes,
                                  growth_rate=growth_rate)
        self.trans3 = TransitionBlockEncoder(inter_planes, inter_planes)
        self.dense4 = _DenseCBAMBlock(num_layers=block_config[2], num_input_features=inter_planes, num_output_features=out_planes,
                                  growth_rate=growth_rate)
        ############# Decoder 0 -32 ##############################
    def forward(self, x):
        input0 = self.pool00(self.conv00(x[:, 0, :, :, :]))
        input1 = self.pool00(self.conv00(x[:, 1, :, :, :]))
        input2 = self.pool00(self.conv00(x[:, 2, :, :, :]))
        input3 = self.pool00(self.conv00(x[:, 3, :, :, :]))
        input4 = self.pool00(self.conv00(x[:, 4, :, :, :]))

        input_cat = torch.cat([input0, input1, input2, input3, input4], 1)
        out_tran = self.conv_tran(input_cat)
        out_dense0 = self.dense1(out_tran)
        out1 = self.dense2(self.trans1(out_dense0))
        out2 = self.dense3(self.trans2(out1))
        out3 = self.dense4(self.trans3(out2))
        return input_cat, out1, out2, out3


class Encoder_4input(nn.Module):
    def __init__(self, pretrain = True, inter_planes=256, out_planes=256, block_config=(6, 4, 4), growth_rate =32):
        super(Encoder_4input, self).__init__()
        ############# Encoder 0 - 256 ##############
        self.conv00 = models.densenet121(pretrained = pretrain).features.conv0
        self.pool00 = models.densenet121().features.pool0
        self.dense00 = models.densenet121(pretrained=pretrain).features.denseblock1
        self.se0 = _SE(1024, 128)
        self.trans1 = TransitionBlockEncoder(256*4, 512)
        self.se1 = _SE(512, 128)
        self.dense2 = _DenseSEBlock(num_layers=block_config[0], num_input_features=512,
                                  num_output_features=inter_planes,
                                  growth_rate=growth_rate)
        self.se2 = _SE(inter_planes, 32)
        self.trans2 = TransitionBlockEncoder(inter_planes, inter_planes)
        ############# Encoder 3 - 32 ##########################
        self.dense3 = _DenseSEBlock(num_layers=block_config[0], num_input_features=inter_planes, num_output_features=inter_planes,
                                  growth_rate=growth_rate)
        self.trans3 = TransitionBlockEncoder(inter_planes, inter_planes)
        self.dense4 = _DenseSEBlock(num_layers=block_config[1], num_input_features=inter_planes, num_output_features=out_planes,
                                  growth_rate=growth_rate)
        ############# Decoder 0 -32 ##############################
    def forward(self, x):
        input0 = self.dense00(self.pool00(self.conv00(x[:, 0, :, :, :])))
        input1 = self.dense00(self.pool00(self.conv00(x[:, 1, :, :, :])))
        input2 = self.dense00(self.pool00(self.conv00(x[:, 2, :, :, :])))
        input3 = self.dense00(self.pool00(self.conv00(x[:, 3, :, :, :])))

        input_cat = torch.cat([input0, input1, input2, input3], 1)
        out0 = self.se0(input_cat)
        out0 = self.trans1(out0)
        out1 = self.se1(out0)
        out1 = self.se2(self.dense2(out1))
        out2 = self.dense3(self.trans2(out1))
        out3 = self.dense4(self.trans3(out2))
        return input_cat, out1, out2, out3



class Encoder2(nn.Module):
    def __init__(self, inter_planes=128, out_planes=256, block_config=(4, 4, 4, 4), growth_rate =32):
        super(Encoder2, self).__init__()
        ############# Encoder 0 - 256 ##############
        self.conv0 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.dense1 = _DenseBlock(num_layers=block_config[0], num_input_features=64, num_output_features=inter_planes,
                                  growth_rate=growth_rate)
        self.trans1 = TransitionBlockEncoder(inter_planes, inter_planes)
        self.dense2 = _DenseBlock(num_layers=block_config[1], num_input_features=inter_planes, num_output_features=inter_planes,
                                  growth_rate=growth_rate)
        self.trans2 = TransitionBlockEncoder(inter_planes, inter_planes)
        ############# Encoder 3 - 32 ##########################
        self.dense3 = _DenseBlock(num_layers=block_config[2], num_input_features=inter_planes, num_output_features=inter_planes,
                                  growth_rate=growth_rate)
        self.trans3 = TransitionBlockEncoder(inter_planes, inter_planes)
        self.dense4 = _DenseBlock(num_layers=block_config[3], num_input_features=inter_planes, num_output_features=out_planes,
                                  growth_rate=growth_rate)
        ############# Decoder 0 -32 ##############################
    def forward(self, x):
        out0 = self.dense1(self.conv0(x))
        out1 = self.dense2(self.trans1(out0))
        out2 = self.dense3(self.trans2(out1))
        out3 = self.dense4(self.trans3(out2))
        return out0, out1, out2, out3

class Decoder2(nn.Module):
    def __init__(self, in_planes=64, inter_planes = 128, out_planes = 32, block_config=(4, 4, 4), growth_rate = 32):
        super(Decoder2, self).__init__()
        ############# Decoder 0 - 256 ##############
        self.TransDecoder0 = TransitionBlockDecoder(in_planes, inter_planes)
        ############# Decoder 1 - 128 ########################
        self.DenseDecoder0 = _DenseBlock(num_layers=block_config[0], num_input_features=inter_planes + 128,
                                         growth_rate=growth_rate, num_output_features=inter_planes)
        self.TransDecoder1 = TransitionBlockDecoder(inter_planes, inter_planes)
        ############# Decoder 2 - 64  ########################
        self.DenseDecoder1 = _DenseBlock(num_layers=block_config[1], num_input_features=inter_planes + 128,
                                         growth_rate=growth_rate, num_output_features=inter_planes)
        self.TransDecoder2= TransitionBlockDecoder(inter_planes, inter_planes)
        ############# Decoder 3 - 32 ##########################
        self.DenseDecoder2 = _DenseBlock(num_layers=block_config[2], num_input_features=inter_planes + 128,
                                         growth_rate=growth_rate, num_output_features=inter_planes)
        ############# Final  ##############################
        self.TransDecoder3 = TransitionBlockDecoder(inter_planes, inter_planes)
        self.DenseDecoder3 = _DenseBlock(num_layers=2, num_input_features=inter_planes, growth_rate=growth_rate, num_output_features=inter_planes)
        self.TransDecoder4 = TransitionBlockDecoder(inter_planes, out_planes)
    def forward(self, x0, x1, x2, x3):
        '''
        :param x0: 256 x 128 x 128
        :param x1: 512 x 64 x 64
        :param x2: 128 x 32 x 32
        :param x3: 256 x 16 x 16
        :return:
        '''
        out3 = self.TransDecoder0(x3)
        out3 = torch.cat([x2, out3], 1)
        out4 = self.TransDecoder1(self.DenseDecoder0(out3))
        out4 = torch.cat([x1, out4], 1)
        out5 = self.TransDecoder2(self.DenseDecoder1(out4))
        out5 = torch.cat([x0, out5], 1)
        out6 = self.TransDecoder3(self.DenseDecoder2(out5))
        out = self.TransDecoder4(self.DenseDecoder3(out6))
        return out



class Decoder(nn.Module):
    def __init__(self, in_planes=64, inter_planes = 128, out_planes = 32, block_config=(4, 4, 4), growth_rate = 32):
        super(Decoder, self).__init__()
        ############# Decoder 0 - 256 ##############
        self.TransDecoder0 = TransitionBlockDecoder(in_planes, 512)
        ############# Decoder 1 - 128 ########################
        num_feat = 512
        self.DenseDecoder0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_feat + inter_planes[0],
                                         growth_rate=growth_rate, num_output_features=(num_feat + block_config[0] * growth_rate)//2)
        num_feat = (num_feat + block_config[0] * growth_rate)//2
        self.TransDecoder1 = TransitionBlockDecoder(num_feat, num_feat)
        ############# Decoder 2 - 64  ########################
        self.DenseDecoder1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_feat + inter_planes[1],
                                         growth_rate=growth_rate, num_output_features=(num_feat + block_config[1] * growth_rate)//2)
        num_feat = (num_feat + block_config[1] * growth_rate)//2
        self.TransDecoder2= TransitionBlockDecoder(num_feat, num_feat)
        ############# Decoder 3 - 32 ##########################
        self.DenseDecoder2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_feat,
                                         growth_rate=growth_rate, num_output_features=(num_feat + block_config[2] * growth_rate)//2)
        num_feat = (num_feat + block_config[2] * growth_rate)//2
        ############# Final  ##############################
        self.TransDecoder3 = TransitionBlockDecoder(num_feat, out_planes, cubic=True)
    def forward(self, x0, x1, x2):
        '''
        :param x0: 256 x 128 x 128
        :param x1: 512 x 64 x 64
        :param x2: 512 x 32 x 32
        :return:
        '''
        out3 = self.TransDecoder0(x2)
        out3 = torch.cat([x1, out3], 1)
        out4 = self.TransDecoder1(self.DenseDecoder0(out3))
        out4 = torch.cat([x0, out4], 1)
        out5 = self.TransDecoder2(self.DenseDecoder1(out4))
        out = self.TransDecoder3(self.DenseDecoder2(out5))
        return out



class DecoderTV(nn.Module):
    def __init__(self, in_planes=64, inter_planes = 128, out_planes = 32, block_config=(4, 4, 4), growth_rate = 32):
        super(DecoderTV, self).__init__()
        ############# Decoder 0 - 256 ##############
        self.TransDecoder0 = TransitionBlockDecoder(in_planes, 512)
        ############# Decoder 1 - 128 ########################
        num_feat = 512
        self.DenseDecoder0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_feat + inter_planes[0], dilation=2,
                                         growth_rate=growth_rate, num_output_features=(num_feat + block_config[0] * growth_rate)//2)
        num_feat = (num_feat + block_config[0] * growth_rate)//2
        self.TransDecoder1 = TransitionBlockDecoder(num_feat, num_feat)
        ############# Decoder 2 - 64  ########################
        self.DenseDecoder1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_feat + inter_planes[1],
                                         growth_rate=growth_rate, num_output_features=(num_feat + block_config[1] * growth_rate)//2)
        num_feat = (num_feat + block_config[1] * growth_rate)//2
        self.TransDecoder2= TransitionBlockDecoder(num_feat, num_feat)
        ############# Decoder 3 - 32 ##########################
        self.DenseDecoder2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_feat,
                                         growth_rate=growth_rate, num_output_features=(num_feat + block_config[2] * growth_rate)//2)
        num_feat = (num_feat + block_config[2] * growth_rate)//2
        ############# Final  ##############################
        self.TransDecoder3 = TransitionBlockDecoder(num_feat, out_planes, cubic=True)
    def forward(self, x0, x1, x2):
        '''
        :param x0: 256 x 128 x 128
        :param x1: 512 x 64 x 64
        :param x2: 512 x 32 x 32
        :return:
        '''
        out30 = self.TransDecoder0(x2)
        out31 = torch.cat([x1, out30], 1)
        out40 = self.TransDecoder1(self.DenseDecoder0(out31))
        out41 = torch.cat([x0, out40], 1)
        out50 = self.TransDecoder2(self.DenseDecoder1(out41))
        out = self.TransDecoder3(self.DenseDecoder2(out50))
        return out, out30, out40, out50



class DecoderSETV(nn.Module):
    def __init__(self, in_planes=64, inter_planes = 128, out_planes = 32, block_config=(4, 4, 4), growth_rate = 32):
        super(DecoderSETV, self).__init__()
        ############# Decoder 0 - 256 ##############
        self.TransDecoder0 = TransitionBlockDecoder(in_planes, 512)
        ############# Decoder 1 - 128 ########################
        num_feat = 512
        self.DenseDecoder0 = _DenseSEBlock(num_layers=block_config[0], num_input_features=num_feat + inter_planes[0],
                                         growth_rate=growth_rate, num_output_features=(num_feat + block_config[0] * growth_rate)//2)
        num_feat = (num_feat + block_config[0] * growth_rate)//2
        self.TransDecoder1 = TransitionBlockDecoder(num_feat, num_feat)
        ############# Decoder 2 - 64  ########################
        self.DenseDecoder1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_feat + inter_planes[1],
                                         growth_rate=growth_rate, num_output_features=(num_feat + block_config[1] * growth_rate)//2)
        num_feat = (num_feat + block_config[1] * growth_rate)//2
        self.TransDecoder2= TransitionBlockDecoder(num_feat, num_feat)
        ############# Decoder 3 - 32 ##########################
        self.DenseDecoder2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_feat,
                                         growth_rate=growth_rate, num_output_features=(num_feat + block_config[2] * growth_rate)//2)
        num_feat = (num_feat + block_config[2] * growth_rate)//2
        ############# Final  ##############################
        self.TransDecoder3 = TransitionBlockDecoder(num_feat, out_planes, cubic=True)
    def forward(self, x0, x1, x2):
        '''
        :param x0: 256 x 128 x 128
        :param x1: 512 x 64 x 64
        :param x2: 512 x 32 x 32
        :return:
        '''
        out30 = self.TransDecoder0(x2)
        out31 = torch.cat([x1, out30], 1)
        out40 = self.TransDecoder1(self.DenseDecoder0(out31))
        out41 = torch.cat([x0, out40], 1)
        out50 = self.TransDecoder2(self.DenseDecoder1(out41))
        out = self.TransDecoder3(self.DenseDecoder2(out50))
        return out, out30, out40, out50





class Decoder_DepEmb(nn.Module):
    def __init__(self, in_planes=64, inter_planes = 128, out_planes = 32, block_config=(4, 4, 4), growth_rate = 32):
        super(Decoder_DepEmb, self).__init__()
        ############# Decoder 0 - 256 ##############
        self.TransDecoder0 = TransitionBlockDecoder(in_planes, inter_planes)
        ############# Decoder 1 - 128 ########################
        self.DenseDecoder0 = _DenseBlock(num_layers=block_config[0], num_input_features=inter_planes + 128,
                                         growth_rate=growth_rate, num_output_features=inter_planes)
        self.TransDecoder1 = TransitionBlockDecoder(inter_planes, inter_planes)
        ############# Decoder 2 - 64  ########################
        self.DenseDecoder1 = _DenseBlock(num_layers=block_config[1], num_input_features=inter_planes + 512,
                                         growth_rate=growth_rate, num_output_features=inter_planes)
        self.TransDecoder2= TransitionBlockDecoder(inter_planes, inter_planes)
        ############# Decoder 3 - 32 ##########################
        self.DenseDecoder2 = _DenseBlock(num_layers=block_config[2], num_input_features=inter_planes + 256,
                                         growth_rate=growth_rate, num_output_features=inter_planes)
        self.DepEmb = depth_embed_module(inter_planes + 256, inter_planes, 32)
        ############# Final  ##############################
        #self.DepEmb0 = depth_embed_module(inter_planes)
        self.TransDecoder3 = TransitionBlockDecoder(inter_planes, inter_planes)
        self.DenseDecoder3 = _DenseBlock(num_layers=2, num_input_features=inter_planes,
                                         growth_rate=growth_rate, num_output_features=inter_planes)

        #self.DepEmb1 = depth_embed_module(inter_planes, num_layers=2)
        self.TransDecoder4 = TransitionBlockDecoder(inter_planes, out_planes)

    def forward(self, x0, x1, x2, x3, d):
        '''
        :param x0: 256 x 128 x 128
        :param x1: 512 x 64 x 64
        :param x2: 128 x 32 x 32
        :param x3: 256 x 16 x 16
        :return:
        '''
        out3 = self.TransDecoder0(x3)
        out3 = torch.cat([x2, out3], 1)
        out4 = self.TransDecoder1(self.DenseDecoder0(out3))
        out4 = torch.cat([x1, out4], 1)
        out5 = self.TransDecoder2(self.DenseDecoder1(out4))
        out5 = torch.cat([x0, out5], 1)
        out6_dense = self.DenseDecoder2(out5)
        out6_unet = self.DepEmb(d[:, :, ::4, ::4], out5)
        out6 = self.TransDecoder3(out6_dense + out6_dense * out6_unet)
        out7 = self.DenseDecoder3(out6)
        out = self.TransDecoder4(out7)
        return out



class Dense(nn.Module):
    def __init__(self, num_classes, pretrain = True, block_config=((4,4), (12,16,4)), growth_rate =32):
        super(Dense, self).__init__()
        ############# First downsampling  ############## 512

        self.encoder = Encoder(pretrain=pretrain, inter_planes=[128, 256, 1024], block_config=block_config[0], growth_rate=growth_rate)
        self.decoder = Decoder(in_planes=1024, inter_planes=[256, 128], out_planes=32, block_config=block_config[1], growth_rate=growth_rate)
        self.conv_out = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=(3,3), stride=(1,1), padding=1)
    def forward(self, x):
        x0, x1, x2 = self.encoder(x)
        out = self.decoder(x0, x1, x2)
        out = self.conv_out(out)
        return out



class DenseTV(nn.Module):
    def __init__(self, num_classes, pretrain = True, block_config=((4,4), (4,4,4)), growth_rate =32):
        super(DenseTV, self).__init__()
        ############# First downsampling  ############## 512

        self.encoder = Encoder(pretrain=pretrain, inter_planes=[128, 256, 512], block_config=block_config[0], growth_rate=growth_rate)
        self.decoder = DecoderTV(in_planes=512, inter_planes=[256, 128], out_planes=32, block_config=block_config[1], growth_rate=growth_rate)
        self.conv_out = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=(3,3), stride=(1,1), padding=1)
    def forward(self, x):
        x0, x1, x2 = self.encoder(x)
        out, out3, out4, out5 = self.decoder(x0, x1, x2)
        out = self.conv_out(out)
        return out, out3, out4, out5



class DenseSETV(nn.Module):
    def __init__(self, num_classes, pretrain = True, block_config=((4,4), (4,4,4)), growth_rate =32):
        super(DenseSETV, self).__init__()
        ############# First downsampling  ############## 512

        self.encoder = EncoderSE(pretrain=pretrain, inter_planes=[128, 256, 512], block_config=block_config[0], growth_rate=growth_rate)
        self.decoder = DecoderSETV(in_planes=512, inter_planes=[256, 128], out_planes=32, block_config=block_config[1], growth_rate=growth_rate)
        self.conv_out = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=(3,3), stride=(1,1), padding=1)
    def forward(self, x):
        x0, x1, x2 = self.encoder(x)
        out, out3, out4, out5 = self.decoder(x0, x1, x2)
        out = self.conv_out(out)
        return out, out3, out4, out5


class Dense_adja(nn.Module):
    def __init__(self, num_classes=4, pretrain = True, inter_planes=256, block_config=((16, 12), (12,16,12)), growth_rate =32):
        super(Dense_adja, self).__init__()
        ############# First downsampling  ############## 512
        #self.encoder_main = Encoder(pretrain=pretrain, inter_planes=inter_planes, out_planes=256, block_config=block_config[0], growth_rate=growth_rate)
        self.encoder_all = Encoder_5input_2branch(pretrain=pretrain, block_config=block_config[0], out_planes=inter_planes,
                                    growth_rate=growth_rate)
        self.decoder = Decoder(in_planes=inter_planes, out_planes=num_classes, inter_planes=128, block_config=block_config[1], growth_rate=growth_rate)
        #self.res1 = _ResidualLayer(input_channels=64, output_channels=64)
        #self.conv_out1 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=(3,3), stride=(1,1), padding=1)

    def forward(self, in_main):
        '''

        :param in_main:  B x 3 x H x W
        :param in_adjacent: B x 5 x 3 x H x W
        :return:
        '''
        x0, x1, x2, x3 = self.encoder_all(in_main)

        decoded = self.decoder(x0, x1, x2)
        #out1 = self.conv_out1(self.res1(decoded))
        #out2 = self.conv_out2(self.res2(decoded))
        #out3 = self.conv_out3(self.res3(decoded))
        #out4 = self.conv_out4(self.res4(decoded))

        return decoded#, out2, out3, out4


class _SE(nn.Module):
    def __init__(self, num_input_features, num_hidden_features):
        super(_SE, self).__init__()
        self.linear1 = nn.Linear(num_input_features, num_hidden_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_hidden_features, num_input_features)
        self.sigmoid = nn.Sigmoid()
    def forward(self, X):
        squeeze = F.avg_pool2d(X, kernel_size=(X.shape[2], X.shape[3]))[:, :, 0, 0]
        excitation = self.sigmoid(self.linear2(self.relu(self.linear1(squeeze))))
        return X + X * (excitation.view(X.shape[0], X.shape[1], 1, 1) - 0.5)

class _ResAttentionBlock(nn.Module):
    def __init__(self, num_input_features, num_layers_trunk=2, num_layers_mask=2, drop_rate=None, memory_efficient=False):
        super(_ResAttentionBlock, self).__init__()
        self.Trunk = nn.Sequential(OrderedDict([]))
        for i in range(num_layers_trunk):
            self.Trunk.add_module('res{}'.format(i), _ResidualLayer(num_input_features, num_input_features))
        self.MaskBlock_up = _SoftMaskBlock(num_layers_mask, num_input_features, drop_rate)
        self.MaskBlock_dn = _SoftMaskBlock(num_layers_mask, num_input_features, drop_rate)

        self.resunit0 = _ResidualLayer(num_input_features, num_input_features, drop_rate)
        self.resunit1 = _ResidualLayer(num_input_features, num_input_features, drop_rate)
        self.resunit2 = _ResidualLayer(num_input_features, num_input_features, drop_rate)
        self.resunit3 = _ResidualLayer(num_input_features, num_input_features, drop_rate)
        self.resunit4 = _ResidualLayer(num_input_features, num_input_features, drop_rate)

        self.relu = nn.ReLU()

    def forward(self, init_features, other_features):
        out_main_res = self.resunit1(self.resunit0(init_features))
        out_other_res = self.resunit3(self.resunit2(other_features))

        out_trunk = self.Trunk(out_main_res)
        out_mask_other = self.MaskBlock_up(out_other_res)

        out_combine = out_main_res + out_other_res
        out = self.resunit4(out_combine)
        out = self.relu(out)
        return out

class _Res4AttentionBlock(nn.Module):
    def __init__(self, num_input_features, num_layers_trunk=4, num_layers_mask=4, drop_rate=None, memory_efficient=False):
        super(_Res4AttentionBlock, self).__init__()
        self.Trunk = nn.Sequential(OrderedDict([]))
        for i in range(num_layers_trunk):
            self.Trunk.add_module('res{}'.format(i), _ResidualLayer(num_input_features, num_input_features))
        self.MaskBlock1 = _SoftMaskBlock(num_layers_mask, num_input_features, drop_rate)
        self.MaskBlock2 = _SoftMaskBlock(num_layers_mask, num_input_features, drop_rate)
        self.MaskBlock3 = _SoftMaskBlock(num_layers_mask, num_input_features, drop_rate)

        self.resunit0 = _ResidualLayer(num_input_features, num_input_features, drop_rate)
        self.resunit1 = _ResidualLayer(num_input_features, num_input_features, drop_rate)
        self.resunit2 = _ResidualLayer(num_input_features, num_input_features, drop_rate)
        self.resunit3 = _ResidualLayer(num_input_features, num_input_features, drop_rate)
        self.resunit4 = _ResidualLayer(num_input_features, num_input_features, drop_rate)
        self.resunit5 = _ResidualLayer(num_input_features, num_input_features, drop_rate)
        self.resunit6 = _ResidualLayer(num_input_features, num_input_features, drop_rate)
        self.resunit7 = _ResidualLayer(num_input_features, num_input_features, drop_rate)

        self.relu = nn.ReLU()

    def forward(self, init_features, other_features0, other_features1, other_features2):
        out_main_res = self.resunit1(self.resunit0(init_features))
        out_other1 = self.resunit3(self.resunit2(other_features0))
        out_other2 = self.resunit5(self.resunit4(other_features1))
        out_other3 = self.resunit7(self.resunit6(other_features2))

        out_trunk = self.Trunk(out_main_res)
        out_mask_other1 = self.MaskBlock1(out_other1)
        out_mask_other2 = self.MaskBlock2(out_other2)
        out_mask_other3 = self.MaskBlock3(out_other3)

        out_combine = out_trunk * (1 + out_mask_other1 + out_mask_other2 + out_mask_other3)
        out = self.resunit4(out_combine)
        out = self.relu(out)
        return out


class CBAM(nn.Module):

    """Convolutional Block Attention Module

    https://eccv2018.org/openaccess/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf

    """

    def __init__(self, in_channels):

        """
        :param in_channels: int

            Number of input channels.

        """

        super().__init__()

        self.CAM = CAM(in_channels)

        self.SAM = SAM()


    def forward(self, input_tensor):

        # Apply channel attention module

        channel_att_map = self.CAM(input_tensor)

        # Perform elementwise multiplication with channel attention map.

        gated_tensor = torch.mul(input_tensor, channel_att_map)  # (bs, c, h, w) x (bs, c, 1, 1)

        # Apply spatial attention module

        spatial_att_map = self.SAM(gated_tensor)

        # Perform elementwise multiplication with spatial attention map.

        refined_tensor = torch.mul(gated_tensor, spatial_att_map)  # (bs, c, h, w) x (bs, 1, h, w)

        return refined_tensor



class CAM(nn.Module):

    """Channel Attention Module

    """

    def __init__(self, in_channels, reduction_ratio=16):

        """
        :param in_channels: int

            Number of input channels.

        :param reduction_ratio: int

            Channels reduction ratio for MLP.
        """

        super().__init__()

        reduced_channels_num = (in_channels // reduction_ratio) if (in_channels > reduction_ratio) else 1

        pointwise_in = nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=reduced_channels_num)

        pointwise_out = nn.Conv2d(kernel_size=1, in_channels=reduced_channels_num, out_channels=in_channels)

        # In the original paper there is a standard MLP with one hidden layer.

        # TODO: try linear layers instead of pointwise convolutions.

        self.MLP = nn.Sequential(pointwise_in, nn.ReLU(), pointwise_out,)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):

        h, w = input_tensor.size(2), input_tensor.size(3)



        # Get (channels, 1, 1) tensor after MaxPool

        max_feat = F.max_pool2d(input_tensor, kernel_size=(h, w), stride=(h, w))

        # Get (channels, 1, 1) tensor after AvgPool

        avg_feat = F.avg_pool2d(input_tensor, kernel_size=(h, w), stride=(h, w))

        # Throw maxpooled and avgpooled features into shared MLP

        max_feat_mlp = self.MLP(max_feat)

        avg_feat_mlp = self.MLP(avg_feat)

        # Get channel attention map of elementwise features sum.

        channel_attention_map = self.sigmoid(max_feat_mlp + avg_feat_mlp)

        return channel_attention_map


class SAM(nn.Module):

    """Spatial Attention Module"""



    def __init__(self, ks=7):

        """



        :param ks: int

            kernel size for spatial conv layer.

        """



        super().__init__()

        self.ks = ks

        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(kernel_size=self.ks, in_channels=2, out_channels=1)



    def _get_padding(self, dim_in, kernel_size, stride):

        """Calculates \'SAME\' padding for conv layer for specific dimension.



        :param dim_in: int

            Size of dimension (height or width).

        :param kernel_size: int

            kernel size used in conv layer.

        :param stride: int

            stride used in conv layer.

        :return: int

            padding

        """



        padding = (stride * (dim_in - 1) - dim_in + kernel_size) // 2

        return padding



    def forward(self, input_tensor):

        c, h, w = input_tensor.size(1), input_tensor.size(2), input_tensor.size(3)


        # Permute input tensor for being able to apply MaxPool and AvgPool along the channel axis

        permuted = input_tensor.view(-1, c, h * w).permute(0,2,1)

        # Get (1, h, w) tensor after MaxPool

        max_feat = F.max_pool1d(permuted, kernel_size=c, stride=c)

        max_feat = max_feat.permute(0,2,1).view(-1, 1, h, w)


        # Get (1, h, w) tensor after AvgPool

        avg_feat = F.avg_pool1d(permuted, kernel_size=c, stride=c)

        avg_feat = avg_feat.permute(0,2,1).view(-1, 1, h, w)



        # Concatenate feature maps along the channel axis, so shape would be (2, h, w)

        concatenated = torch.cat([max_feat, avg_feat], dim=1)

        # Get pad values for SAME padding for conv2d

        h_pad = self._get_padding(concatenated.shape[2], self.ks, 1)

        w_pad = self._get_padding(concatenated.shape[3], self.ks, 1)

        # Get spatial attention map over concatenated features.

        self.conv.padding = (h_pad, w_pad)

        spatial_attention_map = self.sigmoid(

            self.conv(concatenated)

        )

        return spatial_attention_map


class _SoftMaskBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, drop_rate, memory_efficient=False):
        super(_SoftMaskBlock, self).__init__()
        self.down = nn.Sequential(OrderedDict([
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('res0', _ResidualLayer(num_input_features, num_input_features))]))
        self.down_up = nn.Sequential(OrderedDict([
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        for i in range(num_layers):
            self.down_up.add_module('res{}'.format(i), _ResidualLayer(num_input_features, num_input_features))
        self.down_up.add_module('upsample0', nn.UpsamplingBilinear2d(scale_factor=2))
        self.skip = _ResidualLayer(num_input_features, num_input_features)
        self.up = nn.Sequential(OrderedDict([
            ('res{}'.format(num_layers + 2), _ResidualLayer(num_input_features, num_input_features)),
            ('upsample1', nn.UpsamplingBilinear2d(scale_factor=2)),
            ('conv0', nn.Conv2d(num_input_features, num_input_features, kernel_size=1, stride=1,
                                padding=0, bias=False)),
            ('conv1', nn.Conv2d(num_input_features, num_input_features, kernel_size=1, stride=1,
                                padding=0, bias=False)),
            ('sig', nn.Sigmoid())
        ]))

    def forward(self, init_features):
        out_down = self.down(init_features)
        out_downup = self.down_up(out_down)
        out_skip = self.skip(out_downup)
        out_up = self.up(out_downup + out_skip)
        return out_up



class _ResidualLayer(nn.Module):  #@save
    def __init__(self, input_channels, output_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(output_channels, output_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)



class Dense_depth_embed(nn.Module):
    def __init__(self, num_classes, pretrain = True, block_config=((4,4), (12,16,4)), growth_rate =32):
        super(Dense_depth_embed, self).__init__()
        ############# First downsampling  ############## 512
        self.encoder = Encoder(pretrain=pretrain, out_planes=256, block_config=block_config[0], growth_rate=growth_rate)
        self.decoder = Decoder(in_planes=256, out_planes=32, block_config=block_config[1], growth_rate=growth_rate)
        self.conv_out = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=(3,3), stride=(1,1), padding=1)
    def forward(self, x, depth):
        x0, x1, x2, x3 = self.encoder(x)
        out = self.decoder(x0, x1, x2, x3, depth)
        out = self.conv_out(out)
        return out

class depth_embed_module(nn.Module):
    def __init__(self, in_planes, out_planes, inter_planes):
        super(depth_embed_module, self).__init__()
        self.unet = G2(in_planes+1, out_planes, inter_planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, depth, F):
        F_cat = torch.cat([depth, F], 1)
        out = self.sigmoid(self.unet(F_cat))
        return out

class Dense_nopre(nn.Module):
    def __init__(self, num_classes, pretrain = True, block_config=((4,4, 4, 4), (4, 4,4)), growth_rate =32):
        super(Dense_nopre, self).__init__()
        ############# First downsampling  ############## 512
        self.encoder = Encoder2(out_planes=256, block_config=block_config[0], growth_rate=growth_rate)
        self.decoder = Decoder2(in_planes=256, out_planes=32, block_config=block_config[1], growth_rate=growth_rate)
        self.conv_out = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=(3,3), stride=(1,1), padding=1)
    def forward(self, x):
        x0, x1, x2, x3 = self.encoder(x)
        out = self.decoder(x0, x1, x2, x3)
        out = self.conv_out(out)
        return out


class Dense_2dec(nn.Module):
    def __init__(self, num_classes, pretrain = True, block_config=((6,6,6,6), (4,4,4)), growth_rate =32):
        super(Dense_2dec, self).__init__()
        ############# First downsampling  ############## 512

        self.encoder = Encoder2(out_planes=256, block_config=block_config[0], growth_rate=growth_rate)
        self.decoder1 = Decoder2(in_planes=256, out_planes=32, block_config=(6, 4, 4), growth_rate=growth_rate)
        self.conv_out1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3,3), stride=(1,1), padding=1)

        self.decoder2 = Decoder2(in_planes=256, out_planes=32, block_config=block_config[1], growth_rate=growth_rate)
        self.conv_out2 = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=(3, 3), stride=(1, 1),
                                   padding=1)
    def forward(self, x):
        x0, x1, x2, x3 = self.encoder(x)
        out1 = self.decoder1(x0, x1, x2, x3)
        out1 = self.conv_out1(out1)

        out2 = self.decoder2(x0, x1, x2, x3)
        out2 = self.conv_out2(out2)
        return out1, out2


def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block


class G2(nn.Module):
    def __init__(self, input_nc, output_nc, nf):
        super(G2, self).__init__()
        # input is 256 x 256
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))
        # input is 128 x 128
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf * 2, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 64 x 64
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf * 2, nf * 4, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 32
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf * 4, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)
        ## NOTE: decoder
        # input is 32
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer3 = blockUNet(nf * 8, nf * 2, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 64
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer2 = blockUNet(nf* 2 + nf * 4, nf*2, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 128
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = blockUNet(nf * 2 + nf * 2, nf, name, transposed=True, bn=True, relu=True, dropout=False)

        dlayer0 = nn.Sequential()
        dlayer0.add_module('%s_relu' % name, nn.ReLU(inplace=True))
        dlayer0.add_module('%s_tconv' % name, nn.ConvTranspose2d(nf + nf, output_nc, 4, 2, 1, bias=False))

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4

        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1
        self.dlayer0 = dlayer0

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        dout3 = self.dlayer3(out4)
        dout3_out3 = torch.cat([dout3, out3], 1)
        dout2 = self.dlayer2(dout3_out3)
        dout2_out2 = torch.cat([dout2, out2], 1)
        dout1 = self.dlayer1(dout2_out2)
        dout1_out1 = torch.cat([dout1, out1], 1)
        dout0 = self.dlayer0(dout1_out1)

        return dout0


import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder (Downsampling Path)
        self.enc_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.enc_conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc_conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)

        self.enc_conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.enc_conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck_conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bottleneck_conv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder (Upsampling Path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_conv8 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.dec_conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv6 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec_conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.enc_conv2(F.relu(self.enc_conv1(x))))
        x2 = F.relu(self.enc_conv4(F.relu(self.enc_conv3(self.pool1(x1)))))
        x3 = F.relu(self.enc_conv6(F.relu(self.enc_conv5(self.pool2(x2)))))
        x4 = F.relu(self.enc_conv8(F.relu(self.enc_conv7(self.pool3(x3)))))

        # Bottleneck
        x5 = F.relu(self.bottleneck_conv2(F.relu(self.bottleneck_conv1(self.pool4(x4)))))

        # Decoder
        x = self.upconv4(x5)
        x = torch.cat((x, x4), dim=1)
        x = F.relu(self.dec_conv7(F.relu(self.dec_conv8(x))))

        x = self.upconv3(x)
        x = torch.cat((x, x3), dim=1)
        x = F.relu(self.dec_conv5(F.relu(self.dec_conv6(x))))

        x = self.upconv2(x)
        x = torch.cat((x, x2), dim=1)
        x = F.relu(self.dec_conv3(F.relu(self.dec_conv4(x))))

        x = self.upconv1(x)
        x = torch.cat((x, x1), dim=1)
        x = F.relu(self.dec_conv1(F.relu(self.dec_conv2(x))))

        # Final output layer
        x = self.final_conv(x)
        return x

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
