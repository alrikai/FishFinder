import os
import glob
import re

import numpy as np
from PIL import Image

def natural_sort(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

def read_fish_sequence(image_dir, detection_path):
    '''
    gathers the detection bounding boxes and associated frames
    '''
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')), key=natural_sort)
    seq_datapaths = [{'detections': None, 'frame': imgp} for imgp in image_paths]
    bbox_files = sorted(glob.glob(os.path.join(detection_path, '*.txt')), key=natural_sort)
    for bbox_file in bbox_files:
        detections = {}
        #get the detections from the file
        with open(bbox_file, 'rt') as f:
            for bbox_info in f:
                bbox_data = [int(bbox.strip()) for bbox in bbox_info.split(',')]
                inst_id, c1, r1, c2, r2 = bbox_data
                #filter out any erronious boxes (i.e. ones that are < 1px in area)
                bbox_area = abs(c2-c1) * abs(r2-r1)
                if bbox_area > 1:
                    assert(inst_id not in detections)
                    detections[inst_id] = [c1, r1, c2, r2]

        #NOTE: the -1 is to correct for incorrect numbering from the fishlabeler app
        bbox_fnum = int(os.path.basename(bbox_file).split('.')[0]) - 1
        seq_datapaths[bbox_fnum]['detections'] = detections
    return seq_datapaths

def remap_ids(seq_metadata):
    #map each id to a unique ID
    print('remapping inst ids..')
    remapped_metadata = {}
    for seqkey in seq_metadata:
        seq_data = seq_metadata[seqkey]
        id_map_idx = 1
        id_mapping = {}
        remapped_seq_metadata = []
        for frameinfo in seq_data:
            if frameinfo['detections'] is not None and len(frameinfo['detections']) > 0:
                frame_instids = list(frameinfo['detections'].keys())
                active_ids = set(list(id_mapping.keys()))
                frame_idset = set(frame_instids)
                #check if there are any active ids that no longer are present
                inactive_ids = active_ids.difference(frame_idset)
                new_ids = frame_idset.difference(active_ids)
                for new_id in new_ids:
                    #map the current id to the current highest ID
                    id_mapping[new_id] = id_map_idx
                    id_map_idx += 1
                for inactive_id in inactive_ids:
                    id_mapping.pop(inactive_id)

                remapped_bboxes = {}
                for frame_ids in frame_instids:
                    remapped_bboxes[id_mapping[frame_ids]] = frameinfo['detections'][frame_ids]
                remapped_frameinfo = {'frame': frameinfo['frame'], 'detections': remapped_bboxes}
                remapped_seq_metadata.append(remapped_frameinfo)
            else:
                remapped_seq_metadata.append(frameinfo)
                id_mapping = {}
        assert(len(remapped_seq_metadata) == len(seq_data))
        remapped_metadata[seqkey] = remapped_seq_metadata
    return remapped_metadata

def read_fish_dataset(seqlist_path, fishdata_path):
    '''
    NOTE: we assume the layout of
     NRTFish
        |
        |------ Images
        |         |
        |         ----- dset <#1> --> fnum 0 ~ fnum M1
        |         |        ...
        |         |
        |         ----- dset <#N> --> fnum 0 ~ fnum MN
        |
        |--- Annotations
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

    dset_data_list = {}
    with open(seqlist_path, 'rt') as f:
        for seqname in f:
            #strip any newlines, etc
            dset_name = seqname[:-1].strip()

            assert(os.path.exists(fishdata_path))
            images_dir = os.path.join(fishdata_path, 'Images', dset_name)
            detections_dir = os.path.join(fishdata_path, 'Annotations', dset_name)
            dset_data_list[dset_name] = read_fish_sequence(images_dir, detections_dir)

    dset_data_list = remap_ids(dset_data_list)
    return dset_data_list

def clip_bounding_box(bbox, bounds):
    '''
    bbox: bounding box [row low, row high, col low, col high]
    bounds: allowable coordinates [row low, row high, col low, col high]

    returns the bounding box clipped to the allowable coordinate ranges
    '''
    row_bounds = np.clip(bbox[0:2], bounds[0], bounds[1])
    col_bounds = np.clip(bbox[2:4], bounds[2], bounds[3])
    bbox = [*row_bounds, *col_bounds]
    return bbox

def run_annotation_correction(seqlist_path, fishdata_path):
    dset_data_list = {}
    with open(seqlist_path, 'rt') as f:
        for seqname in f:
            #strip any newlines, etc
            dset_name = seqname[:-1].strip()
            seq_metadata_dir = os.path.join(fishdata_path, dset_name)
            assert(os.path.exists(seq_metadata_dir))
            detections_dir = os.path.join(seq_metadata_dir, 'Detections')

            image_paths = sorted(glob.glob(os.path.join(seq_metadata_dir, '*.jpg')), key=natural_sort)
            seq_datapaths = [{'detections': None, 'frame': imgp} for imgp in image_paths]
            bbox_files = sorted(glob.glob(os.path.join(detections_dir, '*.txt')), key=natural_sort)

            hard_errors = []

            #NOTE: the -1 is to correct for incorrect numbering from the fishlabeler app
            for bbox_file in bbox_files:
                bbox_fnum = int(os.path.basename(bbox_file).split('.')[0]) - 1
                detections = {}
                #get the detections from the file
                with open(bbox_file, 'rt') as f:
                    img_fpath = image_paths[bbox_fnum]
                    img = Image.open(img_fpath)
                    width, height = img.size
                    for bbox_info in f:
                        bbox_data = [int(bbox.strip()) for bbox in bbox_info.split(',')]
                        inst_id, c1, r1, c2, r2 = bbox_data
                        #also sort the coordinates, so it is [low, high] order
                        clow = min(c1, c2)
                        chigh = max(c1, c2)
                        rlow = min(r1, r2)
                        rhigh = max(r1, r2)
                        #clip coordinates according to corresponding frame dimensions
                        bbox_coords = clip_bounding_box([clow, chigh, rlow, rhigh], [0, width, 0, height])

                        #if bbox_coords != bbox_data[1::]:
                        #    print ('{} --> {}'.format(bbox_data[1::], bbox_coords))

                        #filter out any erronious boxes (i.e. ones that are < 1px in area)
                        bbox_area = abs(chigh-clow) * abs(rhigh-rlow)
                        if bbox_area > 1:
                            #check if this detection is  duplicate instance ID
                            if inst_id in detections:
                                #check if this is a duplicate of the box already there
                                #-- if so, then ignore it. If not, then it's a user
                                #error and we need user-help to correct it
                                prev_bbox = detections[inst_id]
                                if bbox_coords == prev_bbox:
                                    print('Removing duplicate detection @inst {} file {}'.format(inst_id, bbox_file))
                                else:
                                    hard_error = {'file': bbox_file, 'id': inst_id}
                                    hard_errors.append(hard_error)
                            else:
                                #NOTE: the bbox is now stored [col low, col high, row low, row high]
                                detections[inst_id] = bbox_coords

                seq_datapaths[bbox_fnum]['detections'] = detections
                #NOTE: this is the corrected frame index
                seq_datapaths[bbox_fnum]['fnum'] = bbox_fnum

            if len(hard_errors) > 0:
                print('{}: {} #HARD ERRORS'.format(dset_name, len(hard_errors)))
                for error in hard_errors:
                    print('{} ERROR -- @file {}'.format(error['id'], error['file']))
                print('Correct the above errors, then re-run the correction on this sequence')
                return (False, {})
            else:
                dset_data_list[dset_name] = seq_datapaths
    return (True, dset_data_list)
