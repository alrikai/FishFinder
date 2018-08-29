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

class coco_mapper:
    def __int__ (self, coco_fpath):
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
            if fdir_name not in self.mapper_img_id:
                mapper_img_id[fdir_name] = {}

            mapper_img_id[fdir_name][file_path] = file_globalid
            assert(file_globalid not in mapper_id_img)
            mapper_id_img[file_globalid] = file_path
        return mapper_id_img, mapper_img_id

    def get_fpath_for_id(self, gid):
        return self.coco_mapping[gid]

def run_MHT_cvt(detfeat_datadir, coco_datainfo, detection_info, mht_dir):
    det_proposal_info = utils.load_detectron_detections(coco_datainfo, detection_info)
    id_mapper = coco_mapper(coco_datainfo)
    with open(coco_datainfo, 'rt') as f:
        nhfish_data = json.load(f)

    resolution = det_proposal_info[1]
    #NOTE: these are the frames with detection proposals
    dprop_info = det_proposal_info[0]


    #for each sequence, grab out the corresponding cnn features and save them out
    for seq_name in id_mapper.img_to_id.keys():
        mat_structure = {'bx': [], 'by': [], 'x': [], 'y': [], 'w': [],
                        'h': [], 'r': [], 'fr': [], 'chist': [], 'hog': [], 'cnn': []}

        seq_outgt_filepath = os.path.join(mht_dir, seq_name)
        if not os.path.exists(seq_outgt_filepath):
            os.makedirs(seq_outgt_filepath)
        outgt_filepath = os.path.join(seq_outgt_filepath, 'gt.txt')
        with open(outgt_filepath, 'wt') as gt_fid:
            gt_info = nhfish_data['annotations']
            for gt in gt_info:
                if gt['image_id'] in
                gt_str = make_mht_gt_infostring(gt['image_id'], gt['id'], gt['bbox'])
                gt_fid.write("{}\n".format(gt_str))

        outdet_filepath = os.path.join(mht_dir, 'det.txt')
        with open(outdet_filepath, 'wt') as det_fid:
            for frameid, frameinfo in dprop_info.items():
                for det_prop in frameinfo['detections']:
                    det_str = make_mht_det_infostring(frameid, det_prop['bbox'], det_prop['score'])
                    det_fid.write("{}\n".format(det_str))





        for frameid, frameinfo in dprop_info.items():

            feat_filepath = os.path.join(detfeat_datadir, str(frameid) + '.npz')
            if os.path.exists(feat_filepath):
                cnn_framefeat = np.load(feat_filepath)['data'][None][0]
            else:
                cnn_framefeat = None

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
            if gt['image_id'] in
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
