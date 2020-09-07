import os
import math

import torch
import torch.autograd
import numpy as np

from features.models import deeplab101_FPN as dlab_FPNfeat
import features.utils.davis_utils as dutils

def load_dlab101_features(model_path, num_inputs, target_gpu, mode='train', loadmodel='param'):
    if loadmodel == 'param':
        model = DeeplabFeatures(num_inputs)
        saved_state_dict = torch.load(model_path)
    else:
        model = torch.load(model_path)

    model.float()
    if mode == 'train':
        model.train() # use_global_stats = True
    else:
        model.eval()
    model.cuda(target_gpu)

    if loadmodel == 'param':
        model.load_state_dict(saved_state_dict)
    return model

def make_dlab101_features(model_path, num_inputs, target_gpu, mode='train', loadmodel='param'):

    if loadmodel == 'param':
        model = DeeplabFeatures(num_inputs)
        sstate_dict = torch.load(model_path)
        saved_state_dict = model.resnet_model_surgery(sstate_dict, mode='train')
    else:
        model = torch.load(model_path)

    model.float()
    if mode == 'train':
        model.train() # use_global_stats = True
    else:
        model.eval()
    model.cuda(target_gpu)

    if loadmodel == 'param':
        model.load_resnet_model(saved_state_dict)
    return model

def save_models(model_outdir, feature_model, mask_predictor, guide_window):
    '''
    takes a snapshot of the 3 models, written to the provided output directory
    '''

    if not os.path.exists(model_outdir):
        os.makedirs(model_outdir)
    print ('taking models snapshot @{}'.format(model_outdir))

    feat_snapshot_fname = 'features_model.pth'
    pred_snapshot_fname = 'predictor_model.pth'
    guide_snapshot_fname = 'guide_model.pth'

    feat_msnapshot_path = os.path.join(model_outdir, feat_snapshot_fname)
    torch.save(feature_model.state_dict(), feat_msnapshot_path)
    pred_msnapshot_path = os.path.join(model_outdir, pred_snapshot_fname)
    torch.save(mask_predictor.state_dict(), pred_msnapshot_path)
    if guide_window is not None:
        guide_msnapshot_path = os.path.join(model_outdir, guide_snapshot_fname)
        torch.save(guide_window.state_dict(), guide_msnapshot_path)

    #NOTE: this is for saving out the *entire* model, not just the trainable parameters
    feat_snapshot_fname = 'features_model_all.pth'
    pred_snapshot_fname = 'predictor_model_all.pth'
    guide_snapshot_fname = 'guide_model_all.pth'
    feat_msnapshot_path = os.path.join(model_outdir, feat_snapshot_fname)
    torch.save(feature_model, feat_msnapshot_path)
    pred_msnapshot_path = os.path.join(model_outdir, pred_snapshot_fname)
    torch.save(mask_predictor, pred_msnapshot_path)
    if guide_window is not None:
        guide_msnapshot_path = os.path.join(model_outdir, guide_snapshot_fname)
        torch.save(guide_window, guide_msnapshot_path)

def compute_features(model, image, tgpu):
    '''
    computes the resnet features on the given data (assumed to be numpy arrays)
    image: [batchsz x H x W x 3]
    prev_mask: [batchsz x H x W x 1]
    '''

    net_incpu = image
    #net_incpu = net_incpu.transpose((0,3,1,2))
    #net_in = torch.from_numpy(net_incpu).float()
    net_in = net_incpu.permute(0,3,1,2).float()
    net_in = torch.autograd.Variable(net_in, requires_grad=False).cuda(tgpu)
    featout = model(net_in)
    return featout

def make_models(model_path, model_genparams, runmode):
    #model_path = '../data/deeplab_models/MS_DeepLab_resnet_pretrained_COCO_init.pth'
    feature_model = make_dlab101_features(model_path, model_genparams['features'][0], model_genparams['features'][1], mode='train') #mode='eval'
    return feature_model

def load_models(model_dir, model_metadata, runmode='eval', loadmode='param'):
    '''
    load the features, predictor, and (if not baseline mode), the guide window
    models and return the objects w/ loaded weights

    loadmode: 'param' for loading just the trainable parameters,
              'all' for loading the entire saved model
    '''
    if not os.path.exists(model_dir):
        raise RuntimeError('ERROR: model directory not found')

    if loadmode == 'param':
        feat_snapshot_fname = 'features_model.pth'
    feat_msnapshot_path = os.path.join(model_dir, feat_snapshot_fname)
    num_inputs, target_featgpu = model_metadata['features']
    feature_model = load_dlab101_features(feat_msnapshot_path, num_inputs, target_featgpu, mode=runmode, loadmodel=loadmode)
    return feature_model

def DeeplabFeatures(num_channels=7):
    #use the pyramid arrangement rather than default deeplab
    model = dlab_FPNfeat.FPNResnet(num_channels)
    #model = dlab_feat.MS_Deeplab(dlab_feat.Bottleneck, num_channels)
    return model

def guide_features(model, image, optim_metadata):
    '''
    computes the guide features over the entire frame, returns the deeplab (non-guide
    informed) and guide-tiled features. Both are on GPU optim_metadata['target_featgpu']
    '''
    #compute the features from deeplab-resnet
    dlab_features = compute_features(model, image, optim_metadata['target_featgpu'])

    #if running guide mode, compute the guidance features, if baseline then don't
    return dlab_features[-1]

def process_feature_ROI(features, trk_bbox, guide_dims, roi_dim, optim_metadata, runmode='guide'):
    '''
    takes the full-resolution features, and operates on the ROI as given by the
    tracker to get a fixed-size feature out
    '''
    #map the track ROI from image space to feature space, extract the computed feature block
    #with the specific dimension (in feature space), to then be used for updating the guide
    #hidden state, and computing the mask output
    #NOTE: trk box is expected to be [x1 y1 x2 y2] here --> cvt from [y1y2x1x2] --> [x1y1x2y2]
    roi_trk_box = [trk_bbox[2], trk_bbox[0], trk_bbox[3], trk_bbox[1]]

    #--------------------------------------------------------------------------------
    #TODO: okay, this part is pretty dumb... I need to have the ROI in FEATURE-SPACE,
    #NOT IMAGE SPACE, so I need to do that conversion manually here
    #In theory, I'll get a real tracker written someday that works in feature space
    #and I can blissfully delete this
    featheight, featwidth = features.shape[-2::]
    roi_trk_box_fmap = [tcrd / 8 for tcrd in roi_trk_box]
    roi_trk_box_fmap[0] = max(0, roi_trk_box_fmap[0] - 1)
    roi_trk_box_fmap[1] = max(0, roi_trk_box_fmap[1] - 1)
    roi_trk_box_fmap[2] = min(featwidth-(1/8), roi_trk_box_fmap[2] + 1)
    roi_trk_box_fmap[3] = min(featheight-(1/8), roi_trk_box_fmap[3] + 1)
    #--------------------------------------------------------------------------------

    inst_X_base = dutils.roialign_features(features, roi_trk_box_fmap, guide_dims)
    return inst_X_base
