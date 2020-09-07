import os
import cv2
import glob
from fishfinder.utils import natural_sort

inst_colormap = {
    0: (0, 0, 255),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (128, 0, 128),
    4: (0, 128, 128),
    5: (128, 128, 0),
    6: (0, 255, 255),
    7: (255, 255, 0),
    8: (255, 0, 255),
    8: (0, 0, 0)
}

def annotate_frame(fimg_path, fdet_path, fnum):
    """
    Writes a frame's detection bounding box(es) to the image, and returns it
    fimg_path: directory path to images
    fdet_path: directory path to the annotations
    fnum: the frame number
    """
    img_imagefname = os.path.join(fimg_path, fnum + ".jpg")
    image = cv2.imread(img_imagefname)

    img_detfname = os.path.join(fdet_path, fnum + ".txt")
    if not os.path.exists(img_detfname):
        return image

    #read in the detections and draw them
    with open(img_detfname, 'rt') as det_fid:
        dets = det_fid.readlines()
    detections = [line.rstrip('\n') for line in dets]

    for det in detections:
        bbox_data = [int(c) for c in det.split(",")]
        inst_id = bbox_data[0]
        #x, y, w, h = bbox_data[1::]
        x1, y1, x2, y2 = bbox_data[1::]
        x = x1
        y = y1
        w = abs(x2-x1)
        h = abs(y2-y1)

        bbox_color = inst_colormap[(inst_id % len(inst_colormap))]
        cv2.rectangle(image, (x, y), (x+w, y+h), bbox_color, 2)

    return image

def annotate_video(base_datapath, video_fnum, outpath):
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    video_dpath = os.path.join(base_datapath, video_fnum)
    image_paths = sorted(glob.glob(os.path.join(video_dpath, '*.jpg')), key=natural_sort)
    det_dirpath = os.path.join(video_dpath, "Detections")

    for img_path in image_paths:
        img_dirpath, img_fname = os.path.split(img_path)
        frame_num, ext = os.path.splitext(img_fname)
        assert(ext == ".jpg")
        frame_img = annotate_frame(video_dpath,det_dirpath, frame_num)

        out_fname = os.path.join(outpath, img_fname)
        cv2.imwrite(out_fname, frame_img)

if __name__ == "__main__":
    target_vset = "/home/alrik/Data/NRT/NRTFish-renumber/train.txt"
    with open(target_vset, 'rt') as nrt_fid:
        nrt_vids = nrt_fid.readlines()
    nrt_videos = [line.rstrip('\n') for line in nrt_vids]

    base_dirpath = "/home/alrik/Data/NRT/NRTFish-renumber"
    outpath_basedir = "/media/alrik/HP Portable Drive/backup-HD/nrt-annotated"

    for video_num in nrt_videos:
        outpath = os.path.join(outpath_basedir, video_num)
        annotate_video(base_dirpath, video_num, outpath)
