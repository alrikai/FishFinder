import os
import cv2
import docopt
import json
import numpy as np

import error_visualize as vis

docstr = """Visualize Markup Sampling

Usage:
    markup_vis.py [options]

Options:
    -h, --help                  Print this message
    --metadata_file=<str>       The fileset to visualize from
    --input_dir=<str>           The base directory for the input data
    --output_dir=<str>          The output directory to write visualizations to
    --num_vis=<int>             Number of files to visualize
"""

def visualize_markups(metadata_file, input_dir, output_dir, num_vis):
    os.makedirs(output_dir, exist_ok=True)
    with open(metadata_file) as jfid: 
        train_d = json.load(jfid) 

    nfiles = len(train_d['annotations'])
    idxs = np.random.random_integers(low=0, high=nfiles, size=num_vis)
    for idx in idxs:
        print(f"using index {idx}")
        metadata = train_d['annotations'][idx]
        image_idx = metadata['image_id']
        image_metadata = train_d['images'][image_idx]
        image_path = os.path.join(input_dir, "Images", image_metadata['file_name'])
        image = cv2.imread(image_path)

        other_instances = [td for td in train_d['annotations'] if td['image_id'] == image_idx]
        for bbox_idx, bbox_meta in enumerate(other_instances):
            bbox = bbox_meta['bbox']
            x0, y0, w, h = bbox
            x1, y1 = int(x0 + w), int(y0 + h)
            x0, y0 = int(x0), int(y0)
            image = vis.vis_bbox(image, [x0, y0, x1, y1], bbox_idx)
        
        out_fname = image_metadata['file_name'].replace("/", "_")
        out_fname = f"{idx}-{out_fname}"
        out_filepath = os.path.join(output_dir, out_fname)
        cv2.imwrite(out_filepath, image)

if __name__ == "__main__":
    args = docopt.docopt(docstr, version='v0.1')
    print(args)
    metadata_file = args['--metadata_file']
    input_dir = args['--input_dir']
    output_dir = args['--output_dir']
    num_vis = int(args['--num_vis'])
    visualize_markups(metadata_file, input_dir, output_dir, num_vis)
