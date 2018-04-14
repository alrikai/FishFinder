import os
import glob
import re

def natural_sort(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

def read_fish_sequence(detection_path, image_dir):
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
    fishdata_path/
         |__ [seqnames]/
                |_____ [frames]
                |______detections/
                            |_____ [bbox files]
                |______annotations/ (unused for tracking)
                |______metadata/ (unused for now)
    '''

    dset_data_list = {}
    with open(seqlist_path, 'rt') as f:
        for seqname in f:
            #strip any newlines, etc
            dset_name = seqname[:-1].strip()
            seq_metadata_dir = os.path.join(fishdata_path, dset_name)
            assert(os.path.exists(seq_metadata_dir))
            detections_dir = os.path.join(seq_metadata_dir, 'Detections')
            dset_data_list[dset_name] = read_fish_sequence(detections_dir, seq_metadata_dir)

    dset_data_list = remap_ids(dset_data_list)
    return dset_data_list
