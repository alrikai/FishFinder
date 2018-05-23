import datetime
import json
import os

from PIL import Image
import utils

class NRTFish():
    def __init__(self):
        self.default_dtime = datetime.datetime.now()
        self.info = {
            'year': 2018,
            'version': '0.1',
            'description': 'NRT GroundFish',
            'contributor': 'Alrik Firl',
            'url': 'https://www.overleaf.com/read/szrrfzhkpcbm',
            'date_created': self.default_dtime
        }

        self.license = {
            'id': 0,
            'name': 'MIT',
            'url': 'https://opensource.org/licenses/MIT'
        }

        #for now (and the forseeable future) it'll just be flatfish
        self.supercategory = 'benthos'
        self.category = {
            'id': 1,
            'name': 'flatfish',
            'supercategory': self.supercategory,
        }

        self.frame_idx = 0

    def generate_dataset(self, seqlist_path, data_dir):
        dataset = {
            'info': self.info,
            'images': [],
            'annotations': [],
            'licenses': [self.license]
        }

        data_sequences = utils.read_fish_dataset(seqlist_path, data_dir)

        images = []
        annotations = []

        #load the images & annotations
        for seqkey, seqdata in data_sequences.items():
            base_frame_idx = self.frame_idx
            frame_metadata = [self.generate_image_metadata(frame['frame']) for frame in seqdata]
            images.extend(frame_metadata)

            #NOTE: each bounding box will be a seperate annotation
            for frame_idx, annotation in enumerate(seqdata):
                if annotation['detections'] is not None:
                    annotation_metadata = self.generate_annotation_metadata(base_frame_idx + frame_idx, annotation['detections'])
                    annotations.extend(annotation_metadata)

        dataset['images'].extend(images)
        dataset['annotations'].extend(annotations)

        return dataset

    def generate_annotation_metadata(self, frame_idx, detection_metadata):
        '''
        bbox: dict with keys of instance IDs, and values of the corresponding bounding box
        [upper left x, upper left y, lower right x, lower right y]
        '''
        annotation_metadata = []
        for instid, bbox in detection_metadata.items():
            #change to [upleft x, upleft y, width, height]
            coco_bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
            metadata = {
                'id': instid,
                'image_id': frame_idx,
                'category_id': self.category['id'],
                'segmentation': [],
                'area': coco_bbox[2]*coco_bbox[3],
                'bbox': coco_bbox,
                'iscrowd': 0
            }
            annotation_metadata.append(metadata)
        return annotation_metadata

    def generate_image_metadata(self, image_fname):
        img = Image.open(image_fname)
        width, height = img.size
        #NOTE: we get entire filepaths, and we want to make it a relative path (I think?)
        image_filename = os.sep.join(image_fname.split(os.sep)[-2:])
        #directory arranged as YYYMMDD
        data_date = image_fname.split(os.sep)[-2][:8]
        img_metadata = {
            'id': self.frame_idx,
            'width': height,
            'height': width,
            'file_name': image_filename,
            'license': 0,
            'flickr_url': '',
            'coco_url': '',
            'date_captured': data_date
        }
        self.frame_idx += 1
        return img_metadata


def make_dataset():
    fishdataset = NRTFish()
    seqlist_path = 'data/lists/all.txt'
    data_basedir = '/home/alrik/Data/NRTFishAnnotations_Fix'

    mscoco_fdata = fishdataset.generate_dataset(seqlist_path, data_basedir)
    print('got {} #frames'.format(len(mscoco_fdata['images'])))

    output_file = 'NRT_coco_fish.json'
    with open(output_file, mode='wt') as jfid:
        json.dump(mscoco_fdata, jfid, indent=4, default=str)

if __name__ == "__main__":
    make_dataset()