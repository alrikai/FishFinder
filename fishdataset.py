import os
import glob

import torch
from torch.utils.data import DataLoader
import cv2

import utils

class FishData(torch.utils.data.DataLoader):
    def __init__(self, seqlist_path, data_dir, mode, transform=None, num_epochs=1, det_prop_dir=None):
        if not os.path.exists(data_dir):
            print('Error: data path {} does not exist'.format(data_dir))

        self.detprop_basedir = det_prop_dir
        self.prop_TH = 0.9

        self.data_sequences = utils.read_fish_dataset(seqlist_path, data_dir)
        self.seqkeys = list(self.data_sequences.keys())
        self.num_epochs = num_epochs
        self.mode = mode

        self.seq_idx = 0
        self.epoch_idx = 0
        self.detection_proposals = None
        self.transform = transform

        self.load_sequence()

    def load_sequence(self):
        seq = self.data_sequences[self.sequence_key()]

        seq_detprop_dir = os.path.join(self.detprop_basedir, self.sequence_key())
        if os.path.isdir(seq_detprop_dir):
            det_fnames = sorted(glob.glob(os.path.join(seq_detprop_dir, '*.pkl')), key=utils.natural_sort)
            det_proposals = utils.read_fish_proposals(det_fnames)
            self.detection_proposals = utils.merge_fish_proposals(det_proposals)
        else:
            self.detection_proposals = None

        self.seq_images = [s['frame'] for s in seq]
        self.seq_detection = [s['detections'] for s in seq]
        #TODO: do we want anything else? e.g. to have which frames have detections?

        #TODO: do we want to hav this be arranged by instance? i.e. we could just skip all the
        #frames inbetween detections
        self.seq_dets = [det for det in self.seq_detection if det if not None]

    def next_sequence(self):
        self.seq_idx += 1
        if self.seq_idx >= len(self.seqkeys):
            print('Done with epoch {}'.format(self.epoch_idx))
            self.next_epoch()
        else:
            self.load_sequence()

    def next_epoch(self):
        self.seq_idx = 0
        self.load_sequence()
        self.epoch_idx += 1
        if self.epoch_idx >= self.num_epochs:
            #TODO: do we want to throw a python exception?
            print('Reached end of datasequence')

    def sequence_key(self):
        return self.seqkeys[self.seq_idx]

    def __getitem__(self, index):
        img_path = self.seq_images[index]
        detections = [] if self.seq_detection[index] is None else self.seq_detection[index]
        #TODO: is there a torch function to load images? --> probably
        image = cv2.imread(img_path)
        if self.detection_proposals is not None:
            proposals = self.detection_proposals[index]
            proposals_th = [prop for prop in proposals if prop[-1] >= self.prop_TH]
        else:
            proposals_th = []
        batch = {'image': image, 'roi': detections, 'proposals': proposals_th}
        if self.transform is not None:
            batch = self.transform(batch)
        return batch

    def __len__(self):
        return len(self.seq_images)
