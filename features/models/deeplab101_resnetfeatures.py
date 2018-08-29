import torch
import torch.nn as nn
import torch.autograd
import numpy as np

affine_par = True

#train_bn == False --> all bn parameters are fixed to pretrained model's
train_bn = False #True

def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def get_outshapes(dshape, num_scales=3):
    #assume we are getting the minibatch input's image shape here in order:
    #[batchsz, #channels, height, width]
    assert(len(dshape) == 4 or len(dshape) == 2)
    if len(dshape) == 4:
        input_height = dshape[2]
        input_width = dshape[3]
    else:
        input_height = dshape[0]
        input_width = dshape[1]

    outsize = [
        (int(input_height*0.75)+1, int(input_width*0.75)+1),
        (int(input_height*0.5)+1, int(input_width*0.5)+1),
        (int(outS(input_height)), int(outS(input_width)))]
    return list(outsize[:num_scales])

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  dilation_ = 1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = train_bn
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation_)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = train_bn
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = train_bn
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, NoLabels, indepth=2048):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(indepth, NoLabels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        #TODO: see if the initialization matters here
        for m in self.conv2d_list:
            #m.weight.data.normal_(0, 0.01)
            torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
            return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_channels):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = train_bn
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation__ = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dilation_=dilation__))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        self.c2 = self.layer1(x)
        self.c3 = self.layer2(self.c2)
        self.c4 = self.layer3(self.c3)
        self.c5 = self.layer4(self.c4)
        return self.c5

class SegmentPredict(nn.Module):
    def __init__(self, num_labels, indepth, threshold):
        super(SegmentPredict, self).__init__()
        self.predict = self._make_pred_layer(Classifier_Module, [6,12,18,24], [6,12,18,24], num_labels, indepth)
        self.mask_threshold = threshold
        #performs an adaptive threshold on the mask (though it won't be binarized, it'll be in range [0, 1])
        #self.mask_prediction = nn.Sequential(nn.Sigmoid()) #,
                                             #nn.Conv2d(num_labels, num_labels, (1,1)), #, bias=False),
                                             #nn.ReLU())
        #self.mask_prediction[1].weight.data.normal_(1, 0.01)
        #torch.nn.init.xavier_uniform_(self.mask_prediction[1].weight)

    def _make_pred_layer(self, block, dilation_series, padding_series, num_labels, indepth):
        return block(dilation_series, padding_series, num_labels, indepth)

    def forward(self, outfeat, target_dim):
        outmask = self.predict(outfeat)
        if tuple(outmask.shape[2::]) != target_dim:
            #print('mask res @predictor {} --> {}'.format(tuple(outmask[0].shape[2::]), target_dim))
            outmask = torch.nn.functional.upsample(outmask, size=target_dim, mode='bilinear', align_corners=False)
        return outmask

class MS_Deeplab(nn.Module):
    def __init__(self, block, layers, num_channels):
        super(MS_Deeplab,self).__init__()
        self.Scale = ResNet(block, layers, num_channels)

    def forward(self,x):
        return self.Scale(x)
