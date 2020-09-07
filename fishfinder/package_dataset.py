import os
import shutil
import glob

import docopt
from fishfinder.utils import natural_sort

docstr = """NRTFish Detection Dataset Formatting

Usage:
    package_dataset.py [options]

Options:
    -h, --help                  Print this message
    --datapath=<str>            path to the fish dataset sequences [default: data/fishframes/]
    --outpath=<str>             path to the output directory [default: data/fishframes-fix/]
    --seqlist=<str>             list of sequences to run [default: data/lists/train.txt]
"""

#NOTE: this should be called after the data is corrected, then this script will
#re-package everything into the target dataset filesystem layout, then we can call
#the mscoco formatting after this to generate the requisite mscoco-style json file


def create_output_dirstructure(basepath, seqlist_fpath):
    '''
        NRTFish
        |
        |------ Images
        |         |
        |         ----- dset <#1> --> fnum 0 ~ fnum M1
        |         |        ...
        |         |
        |         ----- dset <#N> --> fnum 0 ~ fnum MN
        |
        |------ Detections
        |         |
        |         ----- dset <#1> --> detection 0 ~ detection K1
        |         |        ...
        |         |
        |         ----- dset <#N> --> detection 0 ~ detection KN
        |
        |---- ImageSets
        |         |
        |         ----- <all.txt | train.txt | test.txt | val.txt>
      __|__
       ___
        _
    '''

    directory_dict = {}
    images_dir = os.path.join(basepath, 'Images')
    annotations_dir = os.path.join(basepath, 'Annotations')
    with open(seqlist_fpath, 'rt') as f:
        for seqname in f:
            #strip any newlines, etc
            dset_name = seqname[:-1].strip()

            seq_imgdir = os.path.join(images_dir, dset_name)
            if not os.path.exists(seq_imgdir):
                os.makedirs(seq_imgdir)
            seq_detdir = os.path.join(annotations_dir, dset_name)
            if not os.path.exists(seq_detdir):
                os.makedirs(seq_detdir)

            directory_dict[dset_name] = {'imgdir': seq_imgdir, 'detdir': seq_detdir}

    lists_dir = os.path.join(basepath, 'ImageSets')
    if not os.path.exists(lists_dir):
        os.makedirs(lists_dir)
    return directory_dict

def repackage_dataset(seqlist_path, fishdata_path, outdata_path, correction_path=None):
    '''
    seqlist_path: path to the text file with the list of sequences to repackage
    fishdata_path: path to the root directory of the NRTFish data
    outdata_path: path to the root directory to write output to
    correction_path: (optional) path to metadata (e.g. if corrected metadata was not done in-place)
    '''

    directory_dict = create_output_dirstructure(outdata_path, seqlist_path)

    if correction_path is None:
        metadata_path = fishdata_path
    else:
        metadata_path = correction_path

    #copy the data from fishdata_path to the file structure in outdata_path
    with open(seqlist_path, 'rt') as f:
        for seqname in f:
            #strip any newlines, etc
            dset_name = seqname[:-1].strip()

            dst_image_dir = directory_dict[dset_name]['imgdir']
            src_image_dir = os.path.join(fishdata_path, dset_name)
            image_paths = sorted(glob.glob(os.path.join(src_image_dir, '*.jpg')), key=natural_sort)
            print ('seq {} --> {} #frames'.format(dset_name, len(image_paths)))
            for image in image_paths:
                shutil.copy2(image, dst_image_dir)

            dst_annotation_dir = directory_dict[dset_name]['detdir']
            src_annotation_dir = os.path.join(metadata_path, dset_name, 'Detections')
            bbox_files = sorted(glob.glob(os.path.join(src_annotation_dir, '*.txt')), key=natural_sort)
            print ('seq {} --> {} #detections'.format(dset_name, len(bbox_files)))
            for bbox_f in bbox_files:
                #TODO: you technically should be splitting this out based on instance ID, I think?
                #or should I just leave that for the MSCOCO part to handle?
                shutil.copy2(bbox_f, dst_annotation_dir)

if __name__ == "__main__":
    args = docopt.docopt(docstr, version='v0.1')
    print(args)
    seqlist_path = args['--seqlist']
    fishdata_path = args['--datapath']
    outdata_path = args['--outpath']

    repackage_dataset(seqlist_path, fishdata_path, outdata_path)
