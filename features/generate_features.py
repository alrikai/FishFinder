import os
import time
import random

import numpy as np

from torch.utils.data import DataLoader

from features.utils import davis_eval_guide_sequence as eval_guideseq
from features.utils import logger

def run_features(fish_dataset, model_path, base_outdir, target_featgpu):
    runmode = 'baseline-features'
    data_outpath = os.path.join(base_outdir, 'eval_output', runmode)
    if not os.path.exists(data_outpath):
        os.makedirs(data_outpath)

    eval_version = 'val' #'test-dev'

    #TODO: needs to be the fish dataset
    fish_dloader = DataLoader(fish_dataset)

    num_classes = 1
    num_inputs = 3

    imgheight, imgwidth = fish_dataset.original_dims
    glogger_logid = str(random.randint(0, (2**32)-1))
    glogger = logger.GuideLogger(base_outdir, glogger_logid, imgheight, imgwidth)

    print ('running feature extraction & pooling...')
    model_loadmetadata = {'features': None}
    model_loadmetadata['features'] = [num_inputs, target_featgpu]

    #remove the sequence gt frame and image (00000_origin.png and 00000_origin.jpg respectively)
    optim_metadata = {'num_classes': num_classes, 'target_featgpu': target_featgpu}
    finetune_params = {'eval_version': eval_version,'mseq_dims': fish_dataset.original_dims}

    #NOTE: this is more useful for debugging to see how the augmentation training outputs look.
    #If this is not of interest, then it is a good idea to turn it off as it slows things down
    eval_guideseq.run_eval_sequence(model_path, model_loadmetadata, fish_dloader,
                                    finetune_params, optim_metadata, glogger, runmode)

    if glogger.logging_on:
        imgheight, imgwidth = fish_dloader.dataset.original_dims
        glogger.set_logdims(imgheight, imgwidth)
        glogger.dump_data(runmode)
        #reset the logging buffers
        glogger.reset_logdata()
