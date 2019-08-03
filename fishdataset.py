import os

import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np

import utils
from features.utils import davis_utils as dutils

class FishData(torch.utils.data.DataLoader):
    def __init__(self, nrt_ddir, coco_dpath, det_dpath, prop_TH=0.2):
        if not os.path.exists(coco_dpath):
            print('Error: data path {} does not exist'.format(coco_dpath))
        if not os.path.exists(det_dpath):
            print('Error: data path {} does not exist'.format(det_dpath))

        self.nrt_ddir = nrt_ddir
        self.prop_TH = prop_TH
        self.data_info, img_dims = utils.load_detectron_detections(coco_dpath, det_dpath, threshold=self.prop_TH)
        self.frame_keys = list(self.data_info.keys())
        self.original_dims = img_dims
        self.seq_idx = 0

        self.target_dims = (101,101)

    def __getitem__(self, index):

        frame_key = self.frame_keys[index]
        frame_data = self.data_info[frame_key]
        image_path = os.path.join(self.nrt_ddir, frame_data['file_path'])
        assert(os.path.exists(image_path))

        image = cv2.imread(image_path)
        det_info = frame_data['detections']
        det_bboxes = [dinfo['bbox'] for dinfo in det_info]
        det_scores = [dinfo['score'] for dinfo in det_info]
        batch = {'image': image, 'roi': det_bboxes, 'score': det_scores, 'id': frame_key}
        return batch

    def __len__(self):
        return len(self.data_info)

    def apply_roi(self, fframes, roi, target_dims):
        '''
        given a set of frames, extract RoIs of the frames s.t. the RoI is equal to
        the target_dims, and is located around the given roi. If the roi is larger
        than target dims, it will be resized to fit within target_dims, and if it
        is smaller, then it will have surrounding area included to reach target_dims
        All input frames are expected to be of dimensions
        [batchsz x height x width x channels], where we only use batchsz == 1
        '''

        #NOTE: ROI stored as [row lowbound, row highbound, col lowbound, col highbound]
        row_roi_bounds = roi[0:2]
        col_roi_bounds = roi[2:4]
        cvt_frames = [isinstance(frm, torch.Tensor) for frm in fframes]
        roi_frames = []
        for idx, frame in enumerate(fframes):
            if cvt_frames[idx]:
                frame = frame.data.numpy()
            assert(len(frame.shape) == 4)

            roi_frame = dutils.apply_ROI_frame(frame[0, ...], row_roi_bounds, col_roi_bounds, target_dims)

            #add the batchsz back in
            roi_frame = np.expand_dims(roi_frame, axis=0)
            if cvt_frames[idx]:
                roi_frame = torch.from_numpy(roi_frame)
            roi_frames.append(roi_frame)
        return roi_frames
