import os
import json

import generate_dataset as gen_dset

def verify_coco_json(coco_data):
    data_keys = list(coco_data.keys())

    target_keys = ['info', 'images', 'annotations', 'licenses']
    for tkey in target_keys:
        assert(tkey in data_keys)

    assert(len(coco_data['images']) == 1)
    image_id = coco_data['images'][0]['id']
    assert(image_id == 0)

    expected_bboxes = [[50, 75, 50, 125],
                       [0, 0, 5, 15],
                       [0, 0, 5, 15],
                       [200, 400, 100, 100]]

    metadata = coco_data['annotations']
    assert(len(metadata) == 4)
    for mid, det in enumerate(metadata):
        assert(det['image_id'] == image_id)
        assert(det['id'] == mid+1)
        assert(det['category_id'] == 1)

        assert(det['bbox'] == expected_bboxes[mid])

def run_cvt_test():
    data_path = 'data/tests/NRTFish'
    seq_tname = '314159265358979_test'

    test_list_fname = 'test_List.txt'
    tlist_fpath = os.path.join(data_path, test_list_fname)
    with open(tlist_fpath, 'wt') as tfid:
        tfid.write('{}\n'.format(seq_tname))

    out_dir = 'data/tests/NRTFish_COCO'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    scratch_path = 'data/tests/NRTFish_scratch'
    if not os.path.exists(scratch_path):
        os.makedirs(scratch_path)
    gen_dset.generate_dataset(tlist_fpath, data_path, scratch_path, out_dir)


    dset_type = os.path.basename(test_list_fname).split('.')[0]
    target_json_file = os.path.join(out_dir, 'nrtfish_' + dset_type + '.json')
    assert(os.path.exists(target_json_file))

    #read the 'dataset' in out_dir and verify correctness
    with open(target_json_file) as fid:
        det_data = json.load(fid)
        verify_coco_json(det_data)
    print('if you reached here, then we cool man')

if __name__ == "__main__":
    run_cvt_test()
