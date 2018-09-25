import os
import json
import sys

import docopt
#from sklearn.decomposition import PCA
import scipy.io
import numpy as np

import utils

docstr = """Generate NRTFish Detection Dataset

Usage:
    generate_dataset.py [options]

Options:
    -h, --help                  Print this message
    --mht_datapath=<str>        path to write results to
    --det_featdir=<str>            path to coco-format directory
    --coco_annotations=<str>     path to coco annotations file
    --coco_detections=<str>     path to coco detection proposals file
"""

def make_mht_infostring(data):
    data_str = ", ".join([str(d) for d in data])
    return data_str

def make_mht_gt_infostring(frame, trkid, bbox):
    '''
    frame, id, bbox-upperleft, bbox-uppertop, bbox width, bbox height
    assumes bbox format in [x, y, width, height] format
    '''
    data = [frame, trkid, bbox[0], bbox[1], bbox[2], bbox[3], 1, -1, -1, -1]
    return make_mht_infostring(data)

def make_mht_det_infostring(frame, bbox, conf):
    data = [frame, -1, bbox[0], bbox[1], bbox[2], bbox[3], conf, -1, -1, -1]
    return make_mht_infostring(data)

class CocoMapper:
    def __init__ (self, coco_fpath):
        self.id_to_img, self.img_to_id = self.load_coco_json(coco_fpath)

    def load_coco_json(self, coco_fpath):
        with open(coco_fpath, 'rt') as coco_fid:
            coco_data = json.load(coco_fid)

        mapper_id_img = {}
        mapper_img_id = {}
        for fdata in coco_data['images']:
            file_globalid = fdata['id']
            file_path = fdata['file_name']
            fdir_name = os.path.dirname(file_path)
            fpath_name = os.path.basename(file_path)
            if fdir_name not in mapper_img_id:
                mapper_img_id[fdir_name] = {}

            mapper_img_id[fdir_name][fpath_name] = file_globalid
            assert(file_globalid not in mapper_id_img)
            mapper_id_img[file_globalid] = file_path
        return mapper_id_img, mapper_img_id

    def get_fpath_for_id(self, gid):
        if gid in self.id_to_img:
            return self.id_to_img[gid]
        else:
            return None

    def get_gid_for_path(self, seq, path_id):
        if path_id in self.img_to_id[seq]:
            return self.img_to_id[seq][path_id]

        else:
            return None

def run_MHT_cvt(detfeat_datadir, coco_datainfo, detection_info, mht_dir):
    det_proposal_info = utils.load_detectron_detections(coco_datainfo, detection_info)
    id_mapper = CocoMapper(coco_datainfo)
    with open(coco_datainfo, 'rt') as f:
        nhfish_data = json.load(f)

    resolution = det_proposal_info[1]
    #NOTE: these are the frames with detection proposals
    dprop_info = det_proposal_info[0]

    fishdata_path = '/home/alrik/Data/NRTFish/val'
    seqlist_path = 'data/lists/val.txt'
    #NOTE: this is the data on disk from the labeling, not in coco format
    nhfish_dset = utils.read_fish_dataset(seqlist_path, fishdata_path)

    #for each sequence, grab out the corresponding cnn features and save them out
    for seq_name in id_mapper.img_to_id.keys():
        seq_outgt_filepath = os.path.join(mht_dir, seq_name)
        if not os.path.exists(seq_outgt_filepath):
            os.makedirs(seq_outgt_filepath)

        gt_bboxinfo = [fd for fd in nhfish_dset[seq_name] if fd['detections'] is not None]
        outgt_filepath = os.path.join(seq_outgt_filepath, 'gt.txt')
        with open(outgt_filepath, 'wt') as gt_fid:
            #gt_info = nhfish_data['annotations']
            for gt in gt_bboxinfo:
                frame_fname = os.path.splitext(os.path.basename(gt['frame']))[0]
                frame_imageid = int(frame_fname)
                for gt_trkid, gt_bbox in gt['detections'].items():
                    #change to [upleft x, upleft y, width, height]
                    gt_bbox_coco = [gt_bbox[0], gt_bbox[1], gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1]]
                    gt_str = make_mht_gt_infostring(frame_imageid,  gt_trkid, gt_bbox_coco)
                    gt_fid.write("{}\n".format(gt_str))

        outdet_filepath = os.path.join(seq_outgt_filepath, 'det.txt')
        with open(outdet_filepath, 'wt') as det_fid:
            for frameid, frameinfo in dprop_info.items():
                for det_prop in frameinfo['detections']:
                    det_str = make_mht_det_infostring(frameid, det_prop['bbox'], det_prop['score'])
                    det_fid.write("{}\n".format(det_str))

        mat_structure = {'bx': [], 'by': [], 'x': [], 'y': [], 'w': [],
                            'h': [], 'r': [], 'fr': [], 'chist': [], 'hog': [], 'cnn': []}

        for frame_idx in range(len(nhfish_dset[seq_name])):
            fpath = nhfish_dset[seq_name][frame_idx]['frame']
            fdir, fname = fpath.split(os.sep)[-2:]
            assert(fdir == seq_name)
            gid = id_mapper.get_gid_for_path(seq_name, fname)

            #NOTE: if gid == None, then the current frame has no annotations, and we skip it
            if gid is None:
                continue

            fnum = int(os.path.splitext(fname)[0])
            feat_fname = str(fnum) + '.npz'
            feat_fname_path = os.path.join(detfeat_datadir, feat_fname)
            if not os.path.exists(feat_fname_path):
                #NOTE: this means that there was a gt bbox, but there was no detection for it (and thus no cnn feature)
                #raise RuntimeError("cnn feature file {} must exist".format(feat_fname_path))
                cnn_framefeat = None
            else:
                cnn_framefeat = np.load(feat_fname_path)['data'][None][0]

            #TODO: we basically want to loop over all the annotation text files in the
            #sequence directory (e.g. at /home/alrik/Data/NRTFish/train), and for each
            #file, get the global id for the coco formatting (i.e. using the
            #id_mapper's img --> id), then grab the cnn feature file for that global id,
            #then save that feature out in the mat structure.
            #We also want to grab the bbox data from each file, and the sequence-specific
            #image id

            for det_idx, det_prop in enumerate(dprop_info[gid]['detections']):
                #TODO: need to append to the structure for each field
                mat_structure['fr'].append(frame_idx)
                mat_structure['bx'].append(det_prop['bbox'][0])
                mat_structure['x'].append(det_prop['bbox'][0])
                mat_structure['by'].append(det_prop['bbox'][1])
                mat_structure['y'].append(det_prop['bbox'][1])
                mat_structure['w'].append(det_prop['bbox'][2])
                mat_structure['h'].append(det_prop['bbox'][3])
                mat_structure['r'].append(det_prop['score'])
                mat_structure['chist'].append([])
                mat_structure['hog'].append([])

                if cnn_framefeat is not None:
                    bbox_feat_data = cnn_framefeat['feature'][det_idx]
                    for cidx in range(len(det_prop['bbox'])):
                        assert(int(bbox_feat_data['bbox'][cidx]) - int(det_prop['bbox'][cidx]) == 0)
                    feat_val = np.squeeze(bbox_feat_data['feat'])
                    mat_structure['cnn'].append(feat_val.flatten())
                else:
                    mat_structure['cnn'].append([])

    feat_fname = os.path.splitext(os.path.basename(coco_datainfo))[0]
    scio_featfile = os.path.join(mht_dir, feat_fname + '.mat')
    scipy.io.savemat(scio_featfile, mat_structure)

def large_mht(detfeat_datadir, coco_datainfo, detection_info, mht_dir):
    det_proposal_info = utils.load_detectron_detections(coco_datainfo, detection_info)
    id_mapper = coco_mapper(coco_datainfo)
    with open(coco_datainfo, 'rt') as f:
        nhfish_data = json.load(f)

    resolution = det_proposal_info[1]
    #NOTE: these are the frames with detection proposals
    dprop_info = det_proposal_info[0]

    outgt_filepath = os.path.join(mht_dir, 'gt.txt')
    with open(outgt_filepath, 'wt') as gt_fid:
        gt_info = nhfish_data['annotations']
        for gt in gt_info:
            gt_str = make_mht_gt_infostring(gt['image_id'], gt['id'], gt['bbox'])
            gt_fid.write("{}\n".format(gt_str))

    outdet_filepath = os.path.join(mht_dir, 'det.txt')
    with open(outdet_filepath, 'wt') as det_fid:
        for frameid, frameinfo in dprop_info.items():
            for det_prop in frameinfo['detections']:
                det_str = make_mht_det_infostring(frameid, det_prop['bbox'], det_prop['score'])
                det_fid.write("{}\n".format(det_str))


    mat_structure = {'bx': [], 'by': [], 'x': [], 'y': [], 'w': [],
                    'h': [], 'r': [], 'fr': [], 'chist': [], 'hog': [], 'cnn': []}
    for det_idx, det_prop in enumerate(frameinfo['detections']):
        #TODO: need to append to the structure for each field
        mat_structure['fr'].append(frameid)

        mat_structure['bx'].append(det_prop['bbox'][0])
        mat_structure['x'].append(det_prop['bbox'][0])
        mat_structure['by'].append(det_prop['bbox'][1])
        mat_structure['y'].append(det_prop['bbox'][1])
        mat_structure['w'].append(det_prop['bbox'][2])
        mat_structure['h'].append(det_prop['bbox'][3])
        mat_structure['r'].append(det_prop['score'])
        mat_structure['chist'].append([])
        mat_structure['hog'].append([])

        if cnn_framefeat is not None:
            bbox_feat_data = cnn_framefeat['feature'][det_idx]
            for cidx in range(len(det_prop['bbox'])):
                assert(int(bbox_feat_data['bbox'][cidx]) - int(det_prop['bbox'][cidx]) == 0)
            feat_val = np.squeeze(bbox_feat_data['feat'])
            mat_structure['cnn'].append(feat_val.flatten())
        else:
            mat_structure['cnn'].append([])

    feat_fname = os.path.splitext(os.path.basename(coco_datainfo))[0]
    scio_featfile = os.path.join(mht_dir, feat_fname + '.mat')
    scipy.io.savemat(scio_featfile, mat_structure)

if __name__ == "__main__":
    args = docopt.docopt(docstr, version='v0.1')
    print(args)

    detfeat_datadir = args['--det_featdir']
    coco_annotation_path = args['--coco_annotations']
    coco_detections_path = args['--coco_detections']
    mht_datadir = args['--mht_datapath']

    run_MHT_cvt(detfeat_datadir, coco_annotation_path, coco_detections_path, mht_datadir)
