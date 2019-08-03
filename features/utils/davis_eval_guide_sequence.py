import os

import torch
import torch.nn

import features.utils.davis_utils as dutils
import features.utils.model_utils as mutils

def save_augoutputs(augout, aug_outdir, imgheight, imgwidth, seqkey, mask_palette):
    if not os.path.exists(aug_outdir):
        os.makedirs(aug_outdir)
    for augidx, auginfo in enumerate(augout):
        dutils.save_training_outputs(auginfo, aug_outdir, augidx, seqkey, imgheight, imgwidth, mask_palette)

def eval_frame(model, frame_params, optim_metadata, apply_roifn):
    '''
    runs the main computation for a frame, and runs backprop on the given frame's results
    operates on the full-resolution frame
    '''

    roidims = frame_params['roidims']
    image = frame_params['mbatch_data']['image']
    bsz, imheight, imwidth = image.shape[:3]
    assert(bsz == 1)

    #perform the feature extraction; X: deeplab, Xguide: tiled guide computed from X
    X = mutils.guide_features(model, image, optim_metadata)

    #TODO: this will be a list of lists, s.t. each bbox is a list of 4 scalars
    det_rois = frame_params['mbatch_data']['roi']
    det_features = []

    #extract features for each bounding box in the frame
    for idx, bbox in enumerate(det_rois):
        trk_bbox = [int(crd.data[0]) for crd in bbox]
        troi_dims = (2, 2)
        score = frame_params['mbatch_data']['score'][idx]
        feats = mutils.process_feature_ROI(X, trk_bbox, troi_dims, roidims, optim_metadata, runmode=frame_params['runmode'])
        featdict = {'feat': feats.cpu().data.numpy(),
                    'bbox': trk_bbox,
                    'score': float(score.data[0])}
        det_features.append(featdict)
        ######################################################################
    return det_features

def run_eval_sequence(model_path, model_loadmetadata, eval_davis_dloader, finetune_params, optim_metadata, glogger, runmode='guide'):
    #feature_model = mutils.load_models(model_path, model_loadmetadata, runmode='eval', loadmode=loadmode)
    feature_model = mutils.make_models(model_path, model_loadmetadata, runmode='eval')
    mseq_gtheight, mseq_gtwidth = finetune_params['mseq_dims']
    #evaluate the fine-tuned model
    feature_model.train()

    #we want to save the results out during eval mode
    glogger.set_loggingmode(True)
    glogger.set_logdims(mseq_gtheight, mseq_gtwidth)
    with torch.no_grad():
        for frame_idx, minibatch in enumerate(eval_davis_dloader):
            if frame_idx % 50 == 0:
                print('running {} of {}'.format(frame_idx, len(eval_davis_dloader)))

            image = minibatch['image']
            bsz, fullheight, fullwidth = image.shape[:3]
            assert(bsz == 1)
            assert(mseq_gtheight == fullheight)
            assert(mseq_gtwidth == fullwidth)

            #NOTE: this is for if we want to use an oracle tracker. Only feasible if using val
            frame_evalparams = {'usegt': True,
                                'mbatch_data': minibatch,
                                'roidims': eval_davis_dloader.dataset.target_dims, #TODO: this has to be the predicted ROI
                                'runmode': runmode}
            trk_feature = eval_frame(feature_model, frame_evalparams, optim_metadata, eval_davis_dloader.dataset.apply_roi)

            #--------------------------------------------------------------------
            #NOTE: to save some cycles, we only prepare the output frames if we are going to be
            #writing them to disk, otherwise we dont bother
            if glogger.logging_on:
                det_id = minibatch['id']
                det_data = {'feature': trk_feature, 'id': int(det_id), 'frame': image.data.numpy()}
                glogger.add_logdata('features', det_data, 'features')
                glogger.dump_data(runmode)
                glogger.reset_logdata()
            ######################################################################
