import os

import torch
from torch.utils.data import DataLoader
import cv2

import utils

class FishData(torch.utils.data.DataLoader):
    def __init__(self, seqlist_path, data_dir, mode, transform=None, num_epochs=1):
        if not os.path.exists(data_dir):
            print('Error: data path {} does not exist'.format(data_dir))

        self.data_sequences = utils.read_fish_dataset(seqlist_path, data_dir)
        self.seqkeys = list(self.data_sequences.keys())
        self.num_epochs = num_epochs
        self.mode = mode
        self.seq_idx = 0
        self.epoch_idx = 0
        self.load_sequence()

    def load_sequence(self):
        seq = self.data_sequences[self.sequence_key()]
        self.seq_images = seq['frames']
        self.seq_detection = seq['detections']
        #TODO: do we want anything else? e.g. to have which frames have detections?

        #TODO: do we want to hav this be arranged by instance? i.e. we could just skip all the
        #frames inbetween detections
        self.seq_dets = [det['detections'] for det in self.seq_detection if det['detections'] if not None]

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
        detections = self.seq_detection[index]
        #TODO: is there a torch function to load images? --> probably
        image = cv2.imread(img_path)
        batch = {'image': image, 'roi': detections}
        if self.transform is not None:
            batch = self.transform(batch)
        return batch

    def __len__(self):
        return len(self.seq_images)
