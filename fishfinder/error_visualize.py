import os

import docopt
import cv2

docstr = """Visualize Errors

Usage:
    error_visualize.py [options]

Options:
    -h, --help                  Print this message
    --imagepath=<str>           path to corresponding image
    --errorpath=<str>           path to bounding box to visualize
    --num_surround=<int>        number of surrounding frames to show [default: 2]
"""

inst_colors = [(255, 0, 0),       #id 0: B
               (0, 255, 0),       #id 1: G
               (0, 0, 255),       #id 2: R
               (255, 255, 255),   #id 3: W
               (0, 255, 255),
               (255, 0, 255),
               (255, 255, 0)]


def vis_bbox(img, bbox, idx, thick=2):
    """Visualizes a bounding box."""
    (x0, y0, x1, y1) = bbox
    #x1, y1 = int(x0 + w), int(y0 + h)
    #x0, y0 = int(x0), int(y0)
    color = inst_colors[idx % len(inst_colors)]
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=thick)
    return img

def visualize_file(img_file, bbox_file):
    bbox_fname = os.path.basename(bbox_file).split('.')[0]
    image = cv2.imread(img_file)
    with open(bbox_file, 'rt') as f:
        bbox_count = 0
        for bbox_info in f:
            bbox_data = [int(bbox.strip()) for bbox in bbox_info.split(',')]
            image = vis_bbox(image, bbox_data[1::], bbox_count)
            bbox_count += 1
    cv2.imwrite(bbox_fname + '.png', image)

def visualize_errors(img_file, bbox_file, num_context=2):
    img_dir = os.path.dirname(img_file)
    box_dir = os.path.dirname(bbox_file)

    bbox_fname = os.path.basename(bbox_file).split('.')[0]
    fname_len = len(bbox_fname)
    for offset in range(-num_context, num_context+1):
        fnum = int(bbox_fname)-offset
        fname_num = str(fnum).zfill(fname_len)
        img_path = os.path.join(img_dir, fname_num + '.jpg')
        box_path = os.path.join(box_dir, fname_num + '.txt')
        visualize_file(img_path, box_path)

if __name__ == "__main__":
    args = docopt.docopt(docstr, version='v0.1')
    print(args)

    img_data = args['--imagepath']
    bbox_data = args['--errorpath']
    ncontext = args['--num_surround']
    visualize_errors(img_data, bbox_data, ncontext)
