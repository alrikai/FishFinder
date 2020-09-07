import os
import shutil
import glob

import docopt
import utils


docstr = """Renumber NRTFish Dataset Annotations

Usage:
    renumber_annotations.py [options]

Options:
    -h, --help                  Print this message
    --datapath=<str>            path to the fish dataset sequences
    --outpath=<str>             path to write corrected metadata to
    --seqlist=<str>             list of sequences to run
"""

def renumber_annotations(annotation_dir, outdir, dset_list):

    with open(dset_list, 'rt') as fid:
        for seq_fname in fid:
            seq = seq_fname.strip()
            seq_annotation_dir = os.path.join(annotation_dir, seq, 'Detections')
            assert(os.path.exists(seq_annotation_dir))
            annotation_paths = sorted(glob.glob(os.path.join(seq_annotation_dir, '*.txt')), key=utils.natural_sort)
            seq_outdir = os.path.join(outdir, seq)
            if not os.path.exists(seq_outdir):
                os.makedirs(seq_outdir)

            for annotation_f in annotation_paths:
                annotation_num, annotation_ext = os.path.splitext(os.path.basename(annotation_f))
                annotation_newnum = str(int(annotation_num)-1).zfill(len(annotation_num))
                out_fname = os.path.join(seq_outdir, annotation_newnum + annotation_ext)
                shutil.copyfile(annotation_f, out_fname)

if __name__ == "__main__":
    args = docopt.docopt(docstr, version='v0.1')
    print(args)

    seqlist_path = args['--seqlist']
    fishdata_path = args['--datapath']
    outdata_path = args['--outpath']
    renumber_annotations(fishdata_path, outdata_path, seqlist_path)
