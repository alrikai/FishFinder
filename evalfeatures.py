import docopt
import time

import fishdataset as fdset
from features import generate_features as featgen

docstr = """NRTFish Tracking-by-Detection

Usage:
    train.py [options]

Options:
    -h, --help                  Print this message
    --modelPath=<str>           Snapshot to load
    --nrtdpath=<str>
    --cocopath=<str>              Evaluation images path prefix [default: data/img/]
    --detpath=<str>
    --outpath=<str>             Output directory to write results to [default: data/output/]
    --gpu0=<int>                GPU number [default: 0]
"""

def compute_features(args):
    fishdata_path = args['--cocopath']
    detproposal_path = args['--detpath']

    base_datadir = args['--nrtdpath']
    base_outdir = args['--outpath']
    modelpath = args['--modelPath']
    tgpu = int(args['--gpu0'])

    dataset = fdset.FishData(base_datadir, fishdata_path, detproposal_path)
    featgen.run_features(dataset, modelpath, base_outdir, tgpu)

if __name__ == "__main__":
    args = docopt.docopt(docstr, version='v0.1')
    print(args)
    start = time.time()

    compute_features(args)

    end = time.time()
    print ('Total time: {} seconds'.format(end-start))
