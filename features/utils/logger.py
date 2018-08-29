import os
import torch
import numpy as np
import PIL
import PIL.Image
import cv2

import features.utils.davis_utils as dutils

class GuideLogger:
    def __init__(self, base_logdir, logid, imgheight, imgwidth):
        '''
        base_logdir: the base directory to use for logging
        id: random id to make sure things are unique
        imgheight, imgwidth: frame dimensions (for writing out the image ROIs)
        '''

        target_dir = os.path.join(base_logdir, logid)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        self.logdir = target_dir
        self.logfile_path = os.path.join(self.logdir, logid + '.txt')

        self.logging_on = True
        self.set_logdims(imgheight, imgwidth)
        self.reset_logdata()

    def set_loggingmode(self, mode):
        '''
        For now, the modes are ON (True) or OFF (False)
        '''
        self.logging_on = mode

    def set_logdims(self, imgheight, imgwidth):
        '''
        although most sequences will be the same dimension, some will not be
        '''
        self.imgheight = imgheight
        self.imgwidth = imgwidth

    def add_textlog(self, text):
        #TODO: need to decide if I want to have different logging levels, i.e. I think I always
        #want to log the text, but it's  bit hard to say for sure?
        #if self.logging_on:
        self.logtext.append(text)
        print(text)

    def add_framelog(self, logname, logtype='mask'):
        if self.logging_on:
            if logname not in self.logdata:
                self.logdata[logname] = []
            if logname not in self.logtypes:
                self.logtypes[logname] = logtype

    def add_logdata(self, logname, data, logtype='mask'):
        if self.logging_on:
            if logname not in self.logdata:
                self.add_framelog(logname, logtype=logtype)

            self.logdata[logname].append(data)

    def dump_text(self):
        with open(self.logfile_path, 'ta') as fid:
            for textline in self.logtext:
                fid.write('{}\n'.format(textline))
        self.logtext = []

    def dump_data(self, sdir_path):

        #NOTE: I think we always want to log out the text (for now)
        self.dump_text()

        if self.logging_on:
            for logdata_name, logframedata in self.logdata.items():
                subdir = os.path.join(self.logdir, sdir_path, logdata_name)
                if not os.path.exists(subdir):
                    os.makedirs(subdir)
                if self.logtypes[logdata_name] == 'features':
                    dutils.save_seqoutputs_features(logframedata, subdir)
            #reset all the data, since we just got done logging it
            self.reset_logdata()


    def reset_logdata(self):
        self.logdata = {}
        self.logtypes = {}
        self.logtext = []

    def remap_grey(self, greymask, roi, run_rescale=True):
        '''
        helper function for remapping greyscale ROI images back to full-resolution
        It is assumed that all greyscale frames being logged that require this will
        have already called it before adding the frames to be logged out
        '''

        if isinstance(greymask, torch.Tensor):
            if greymask.requires_grad:
                greymask = greymask.detach().cpu().numpy()
            else:
                greymask = greymask.cpu().numpy()
        greymask = np.squeeze(greymask)

        if run_rescale:
            rescale_div = np.ptp(greymask)
            if rescale_div == 0:
                rescale_mask = greymask
            else:
                rescale_mask = 255 * (greymask - np.min(greymask)) / rescale_div
        else:
            rescale_mask = greymask
        rescale_mask = rescale_mask.astype(np.int32)

        if not dutils.is_RoI_valid(roi, [self.imgheight, self.imgwidth]):
            roi = [0, self.imgheight, 0, self.imgwidth]
        remap_gmask = dutils.remap_mask_roi(rescale_mask, roi, (self.imgheight, self.imgwidth))
        return remap_gmask

    def generate_roimask(self, segmask, original_roi, mask_palette):

        if isinstance(segmask, torch.Tensor):
            if segmask.requires_grad:
                segmask = segmask.detach().cpu().numpy()
            else:
                segmask = segmask.cpu().numpy()
        segmask = np.squeeze(segmask).astype(np.int32)

        out_dims = (self.imgheight, self.imgwidth)
        #merge the per-instance masks
        if not dutils.is_RoI_valid(original_roi, out_dims):
            original_roi = [0, out_dims[0], 0, out_dims[1]]
        outmask = dutils.remap_mask_roi(segmask, original_roi, out_dims)
        pil_mask = PIL.Image.fromarray(outmask)
        if mask_palette is not None:
            pil_mask.putpalette(mask_palette)

        outmask_wroi = outmask.copy()
        #NOTE: ROI is stored [row_low row_high, col_low, col_high]. OpenCV orders
        #everything as (col, row) however, so I need to re-order the coordinates
        pt_A = (original_roi[2], original_roi[0])
        pt_B = (original_roi[3], original_roi[1])
        cv2.rectangle(outmask_wroi, pt_A, pt_B, 255, 2)
        outmask_wroi_pil = PIL.Image.fromarray(outmask_wroi)
        if mask_palette is not None:
            outmask_wroi_pil.putpalette(mask_palette)
        return pil_mask, outmask_wroi_pil


