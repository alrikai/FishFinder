import docopt
import correct_detections as fixdet
import package_dataset as packdset
import convert_to_mscoco as cocodset

docstr = """Generate NRTFish Detection Dataset

Usage:
    generate_dataset.py [options]

Options:
    -h, --help                  Print this message
    --datapath=<str>            path to the fish dataset sequences (to be modified in place)
    --coco_outpath=<str>        path to the output mscoco directory
    --seqlist=<str>             list of sequences to run
"""

args = docopt.docopt(docstr, version='v0.1')
print(args)

seqlist_path = args['--seqlist']
fishdata_path = args['--datapath']
coco_outdata_path = args['--coco_outpath']


def generate_dataset():

    #fix any errors (user or otherwise) that slip through the labeling, if any hard
    #errors, exit and have user fix
    fixed_det = fixdet.correct_detections(seqlist_path, fishdata_path, fishdata_path)
    if not fixed_det:
        return False

    #re-arrange the dataset files to conform to dataset filesystem layout
    packdset.repackage_dataset(seqlist_path, fishdata_path, coco_outdata_path)
    #make the mscoco-style json file
    cocodset.make_dataset(seqlist_path, coco_outdata_path)

if __name__ == "__main__":
    generate_dataset()
