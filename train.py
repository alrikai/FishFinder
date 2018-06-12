import docopt

import utils
import fishdataset as fdset

from torch.utils.data import DataLoader

docstr = """NRTFish Tracking-by-Detection

Usage:
    train.py [options]

Options:
    -h, --help                  Print this message
    --datapath=<str>            path to the fish dataset sequences [default: data/fishframes/]
    --seqlist=<str>             list of sequences to run [default: data/lists/train.txt]
    --detpath=<str>             list of sequences to run
    --gpu0=<int>                GPU number [default: 0]
"""

args = docopt.docopt(docstr, version='v0.1')
print(args)

seqlist_path = args['--seqlist']
fishdata_path = args['--datapath']
detproposal_path = args['--detpath']

num_epochs = 1
dataset = fdset.FishData(seqlist_path, fishdata_path, 'train', det_prop_dir=detproposal_path)
fish_dloader = DataLoader(dataset)

for epoch_idx in range(num_epochs):
    print ('epoch: {} / {} starting'.format(epoch_idx, num_epochs))
    for frame_idx, fishbatch in enumerate(fish_dloader):
        if frame_idx == 0:
            dtks = fish_dloader.dataset.seq_dets
            print ('sequence: {} --> {} #detections, proposals: {}'.format(fish_dloader.dataset.sequence_key(), len(dtks), fishbatch['proposals']))
