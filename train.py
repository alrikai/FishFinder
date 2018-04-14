import docopt

import utils

docstr = """NRTFish Tracking-by-Detection

Usage:
    train.py [options]

Options:
    -h, --help                  Print this message
    --datapath=<str>            path to the fish dataset sequences [default: data/fishframes/]
    --seqlist=<str>             list of sequences to run [default: data/lists/train.txt]
    --gpu0=<int>                GPU number [default: 0]
"""

args = docopt.docopt(docstr, version='v0.1')
print(args)

seqlist_path = args['--seqlist']
fishdata_path = args['--datapath']

datalist = utils.read_fish_dataset(seqlist_path, fishdata_path)


