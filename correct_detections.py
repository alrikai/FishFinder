import os
import docopt
import utils

docstr = """NRTFish Detection Bounding Box Correction

Usage:
    correct_detections.py [options]

Options:
    -h, --help                  Print this message
    --datapath=<str>            path to the fish dataset sequences [default: data/fishframes/]
    --outpath=<str>             path to the output directory [default: data/fishframes-fix/]
    --seqlist=<str>             list of sequences to run [default: data/lists/train.txt]
"""

args = docopt.docopt(docstr, version='v0.1')
print(args)

seqlist_path = args['--seqlist']
fishdata_path = args['--datapath']
outdata_path = args['--outpath']

is_valid, dset_data_list = utils.run_annotation_correction(seqlist_path, fishdata_path)
if is_valid:
    for seqkey, seqdata in dset_data_list.items():
        seq_outdir = os.path.join(outdata_path, seqkey, 'Detections')
        if not os.path.exists(seq_outdir):
            os.makedirs(seq_outdir)

        #means there were no hard errors, and hence we are all good. Need to write out the results
        #to the output directory
        seq_dets = [det for det in seqdata if det['detections'] if not None]
        for sdet in seq_dets:
            det_fname = str(sdet['fnum']).zfill(6) + '.txt'
            det_outpath = os.path.join(seq_outdir, det_fname)
            with open(det_outpath, 'wt') as fout:
                for instid, bbox in sdet['detections'].items():
                    clow, chigh, rlow, rhigh = bbox
                    fout.write('{}, {}, {}, {}, {}\n'.format(instid, clow, rlow, chigh, rhigh))
else:
    print('Fix aforementioned errors and re-run the correction')

#TODO: look for:
# - invalid bounding boxes (i.e clip negative coordinates, ones out of bounds)
# - duplicate detections (sometimes there are just 2 ientical detections)
# - erronious small detections (i.e. ones that have 0 area, just from a stray
# mouse-click)
# -
