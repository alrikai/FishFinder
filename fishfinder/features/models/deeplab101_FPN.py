import torch
import torch.nn as nn
import numpy as np

import models.deeplab101_resnetfeatures as dlabfeat

class FPNResnetiFeatures(nn.Module):
    def __init__(self, block, layers, num_channels):
        super(FPNResnetiFeatures, self).__init__()
        self.deeplab = dlabfeat.MS_Deeplab(block, layers, num_channels)
        self.depth = 256

        self.c2_module = nn.ModuleList([nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(256, self.depth, kernel_size=1),
                                     nn.Conv2d(self.depth, self.depth, kernel_size=3, padding=1, bias=True)])

        self.c3_module = nn.ModuleList([nn.Conv2d(512, self.depth, kernel_size=1),
                                     nn.Conv2d(self.depth, self.depth, kernel_size=3, padding=1, bias=True)])

        self.c4_module = nn.ModuleList([nn.Conv2d(1024, self.depth, kernel_size=1),
                                     nn.Conv2d(self.depth, self.depth, kernel_size=3, padding=1, bias=True)])

        self.c5 = nn.Conv2d(2048, self.depth, kernel_size=1)


        self.fpn = nn.ModuleList([self.c2_module, self.c3_module, self.c4_module])
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            else:
                #print('not sure what to do with {}'.format(type(m)))
                pass

    def forward(self, infeatures):
        botup_feats = self.deeplab(infeatures)
        res_feats = [self.deeplab.Scale.c2, self.deeplab.Scale.c3, self.deeplab.Scale.c4]

        #for top-down, start at highest feature index (and reduce feature depth)
        feat_pyramid = [self.c5(botup_feats)]
        for pidx in range(len(self.fpn)-1, -1, -1):
            lateral_feat = self.fpn[pidx][0](res_feats[pidx])
            if pidx == 0:
                featmap_lvl = self.fpn[pidx][1](feat_pyramid[-1]) + lateral_feat
            else:
                featmap_lvl = feat_pyramid[-1] + lateral_feat
            featmap_lvl = self.fpn[pidx][-1](featmap_lvl)
            feat_pyramid.append(featmap_lvl)
        return feat_pyramid[1::]

class FPNResnet(nn.Module):
    def __init__(self, num_channels):
        super(FPNResnet,self).__init__()
        self.num_inputs = num_channels
        self.features = FPNResnetiFeatures(dlabfeat.Bottleneck, [3, 4, 23, 3], num_channels)

    def get_trainable(self):
        '''
        return the list of trainable parameters here
        '''
        return filter(lambda p:p.requires_grad, self.features.parameters())

    def resnet_model_surgery(self, saved_state_dict, mode='train'):
        '''
        TODO: need to load the MS COCO pre-trained resent weights here, and initialize
        the FPN parts seperately
        '''

        inlayer_key = 'Scale.conv1.weight'
        inconv_shape = saved_state_dict[inlayer_key].cpu().numpy().shape
        del_keys = []
        for lkey in saved_state_dict:
            i_parts = lkey.split('.')
            if i_parts[1]=='layer5':
                del_keys.append(lkey)
            elif lkey == inlayer_key:
                conv_shape = list(saved_state_dict[lkey].shape)
                if mode == 'train':
                    pretrain_ch = min(conv_shape[1], self.num_inputs)
                    conv_shape[1] = self.num_inputs
                    #TODO: if we are loading a trained model, then we need to handle the CPU <--> GPU aspects here
                    new_convweights = torch.FloatTensor(*conv_shape)
                    new_convweights[:, :pretrain_ch, ...] = saved_state_dict[lkey][:, :pretrain_ch, ...]
                    mean_filter = torch.Tensor(np.mean(saved_state_dict[lkey].cpu().numpy(), axis=1)) #, keepdims=True))
                    for fidx in range(pretrain_ch, self.num_inputs):
                        new_convweights[:, fidx, ...] = mean_filter
                    saved_state_dict[lkey] = new_convweights
                else:
                    if inconv_shape[1] != self.num_inputs:
                        assert(inconv_shape[1] > self.num_inputs)
                        conv_shape = list(inconv_shape)
                        conv_shape[1] = self.num_inputs
                        new_convweights = torch.FloatTensor(*conv_shape)
                        new_convweights[:, :self.num_inputs, ...] = saved_state_dict[lkey][:, :self.num_inputs, ...]
                        saved_state_dict[lkey] = new_convweights
        for delk in del_keys:
            del saved_state_dict[delk]
        return saved_state_dict

    def load_resnet_model(self, saved_state_dict):
        self.features.deeplab.load_state_dict(saved_state_dict)

    def forward(self, x):
        return self.features(x)
