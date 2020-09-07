import glob
import os
import re
import math
import json

import numpy as np
import cv2
import PIL
import PIL.Image

import torch
from torchvision import transforms

from roi_align.roi_align import RoIAlign

def natural_sort(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

def save_seqoutputs_greyscale(outputs, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fnames = [str(fidx).zfill(5)+'.png' for fidx in range(len(outputs))]
    for frameidx, fname in enumerate(fnames):
        outpath = os.path.join(outdir, fname)
        cv2.imwrite(outpath, outputs[frameidx])

def save_seqoutputs_pil(outputs, outdir, height, width):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fnames = [str(fidx).zfill(5)+'.png' for fidx in range(len(outputs))]
    for frameidx, fname in enumerate(fnames):
        outpath = os.path.join(outdir, fname)
        mask_data_pil = np.squeeze(np.asarray(outputs[frameidx]))
        mask_data = mask_data_pil.copy()
        outframe = cv2.resize(mask_data, (width, height), interpolation=cv2.INTER_NEAREST)
        pil_mask = PIL.Image.fromarray(outframe.astype(np.uint8))
        mask_palette = outputs[frameidx].getpalette()
        if mask_palette is not None:
            pil_mask.putpalette(mask_palette)
        pil_mask.save(outpath)

def read_flo_file(file_path):
    """
    reads a flo file, it is for little endian architectures,
    first slice, i.e. data2D[:,:,0], is horizontal displacements
    second slice, i.e. data2D[:,:,1], is vertical displacements
    """
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic[0]:
            print('Magic number incorrect. Invalid .flo file: {}'.format(file_path))
            raise  Exception('Magic incorrect: %s !' % file_path)
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2*w*h)
            data2D = np.reshape(data, (h, w, 2), order='C')
            return data2D

def write_flo_file(file_path, data2D):
    """
    writes a flo file, it is for little endian architectures,
    first slice, i.e. data2D[:,:,0], is horizontal displacements
    second slice, i.e. data2D[:,:,1], is vertical displacements
    """
    with open(file_path, 'wb') as f:
        magic = np.array(202021.25, dtype='float32')
        magic.tofile(f)
        h = np.array(data2D.shape[0], dtype='int32')
        w = np.array(data2D.shape[1], dtype='int32')
        w.tofile(f)
        h.tofile(f)
        data2D.astype('float32').tofile(f);

def read_davis_datasets_singleinst(path_to_imagelist, path_to_imgdir, path_to_maskdir, path_to_flowdir):
    '''
    grab the file paths for the given individual DAVIS imagesets -- namely, return
    a dictionary with the dataset name as the key, and a list of image, mask, fwd
    and inv optial flow paths (in sorted, ascending order), with each sequence split
    into individual instances (with the specific instance id stored in the 'id' key)
    '''
    dset_data_list = {}
    has_oflow = path_to_flowdir is not None and os.path.isdir(path_to_flowdir)
    with open(path_to_imagelist, 'rt') as f:
        for line in f:
            dset_name = line[:-1].strip()
            image_dir = os.path.join(path_to_imgdir, dset_name)
            mask_dir = os.path.join(path_to_maskdir, dset_name)
            img_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')), key=natural_sort)
            mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.png')), key=natural_sort)
            #flesh out the mask files for eval sequences (mask path == None, since they don't exist)
            if len(mask_files) == 1:
                seqlen_diff = len(img_files) - 1
                mask_files.extend(seqlen_diff*[None])
            assert(len(mask_files) == len(img_files))

            if has_oflow:
                #my augmentation data is stored in a different format from the DAVIS stuff, which makes
                #my optical flow stuff require 2 seperate paths for fws and inv
                if isinstance(path_to_flowdir, dict):
                    fwdflo_dir = os.path.join(path_to_flowdir['fwd'], dset_name)
                    invflo_dir = os.path.join(path_to_flowdir['inv'], dset_name)
                    #TODO: this is a hack... but since I also include the gt mask in the augmentation
                    #dataset, I want to just delete it here, since it's not part of any augmentation triplet
                    #the hack is that I assume if the flow input is a dict, that it's loading augdata
                    assert('origin' in img_files[-1])
                    img_files.pop()
                    assert('origin' in mask_files[-1])
                    mask_files.pop()
                else:
                    fwdflo_dir = os.path.join(path_to_flowdir, 'fwd', dset_name)
                    invflo_dir = os.path.join(path_to_flowdir, 'inv', dset_name)

                fwdflo_files = sorted(glob.glob(os.path.join(fwdflo_dir, '*.flo')), key=natural_sort)
                invflo_files = sorted(glob.glob(os.path.join(invflo_dir, '*.flo')), key=natural_sort)
                assert(len(fwdflo_files) == len(invflo_files))
            else:
                fwdflo_files = []
                invflo_files = []

            #split off the instances...
            mask_data_pil = PIL.Image.open(mask_files[0])
            mask_data = np.asarray(mask_data_pil)
            gt_temp = mask_data.copy()
            num_inst = len(np.unique(gt_temp))-1
            #make N datasets from the N instances
            for inst_id in range(num_inst):
                dset_inst_name = dset_name + '-' + str(inst_id+1)
                dset_data_list[dset_inst_name] = {'img': img_files, 'mask': mask_files,
                                                  'fwdflo': fwdflo_files, 'invflo': invflo_files, 'id': inst_id+1}
    return dset_data_list

def get_data_from_sequence_davis_singleinst(indata, davis_scale='480p'):
    '''
    given the input paths (as from read_davis_datasets_singleinst), loads the respective
    inputs (i.e. mask, image, optical flows) for the given instance
    '''
    #get the #instances from the 0th frame
    mask_data_pil = PIL.Image.open(indata['mask'][0])
    gt_palette = mask_data_pil.getpalette()
    mask_data_ro = np.asarray(mask_data_pil)
    mask_data = mask_data_ro.copy()
    h_dim, w_dim = mask_data.shape[:2]

    #get the instance ID to use for this sequence
    inst_id = indata['id']
    #since we seperate out each instance, we just treat each instance as a new dataset
    num_frames = len(indata['img'])
    seq_imgs = [None for _ in range(num_frames)]
    seq_masks = [None for _ in range(num_frames)]
    fwd_flow = []
    inv_flow = []
    for frame_idx in range(num_frames):
        img_temp = cv2.imread(indata['img'][frame_idx]).astype(float)
        img_temp[:,:,0] = img_temp[:,:,0] - 104.008
        img_temp[:,:,1] = img_temp[:,:,1] - 116.669
        img_temp[:,:,2] = img_temp[:,:,2] - 122.675
        if len(img_temp.shape) < 3:
            img_temp = np.expand_dims(img_temp, axis=-1)
        seq_imgs[frame_idx] = img_temp

        #NOTE: there will always be 1 less optical flow frame, as it requires 2
        #regular frames to compute 1 optical flow frame
        if frame_idx < len(indata['fwdflo']):
            fwd_flow_frame = read_flo_file(indata['fwdflo'][frame_idx])
            inv_flow_frame = read_flo_file(indata['invflo'][frame_idx])
            fwd_flow.append(fwd_flow_frame)
            inv_flow.append(inv_flow_frame)

        mask_path = indata['mask'][frame_idx]
        if mask_path is not None:
            mask_data_pil = PIL.Image.open(mask_path)
            #this is (often?) read-only -- so just make a copy
            mask_data_ro = np.asarray(mask_data_pil)
            mask_data = mask_data_ro.copy()
            #for num_inst, need to duplicate the images, and seperate the masks
            inst_loc = np.where(mask_data == inst_id)
            inst_gt = np.zeros_like(mask_data)
            inst_gt[inst_loc] = 1

            if len(inst_gt.shape) < 3:
                inst_gt = np.expand_dims(inst_gt, axis=-1)
            seq_masks[frame_idx] = inst_gt

    #insert a 0th inv flow, and an Nth fwd flow (i.e. the -1th frame is the same
    #as the 0th frame, the N+1th frame is the same as the Nth frame)
    has_flow = len(fwd_flow) > 0 and len(inv_flow) > 0
    if has_flow:
        nth_fwd_flow = np.zeros_like(fwd_flow[-1])
        fwd_flow.append(nth_fwd_flow)
        initial_inv_flow = np.zeros_like(inv_flow[-1])
        inv_flow.insert(0, initial_inv_flow)
        assert(len(fwd_flow) == num_frames and len(inv_flow) == num_frames)
    return seq_imgs, seq_masks, gt_palette, fwd_flow, inv_flow

def resize_label_mask(mask, size):
    new_dims = (size[0], size[1], mask.shape[-1])
    mask_resized = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST).astype(float)
    return mask_resized.reshape(new_dims)

def resize_label_batch(mask, size):
    new_dims = (size[0], size[1], 1, mask.shape[3])
    mask_resized = np.zeros(new_dims, dtype=float)
    for frame_idx in range(mask.shape[-1]):
        mask_resized[..., 0, frame_idx] = cv2.resize(mask[..., frame_idx], (size[1], size[0]), interpolation=cv2.INTER_NEAREST).astype(float)
    return mask_resized

def lr_poly(base_lr, iter,max_epoch,power):
    return base_lr*((1-float(iter)/max_epoch)**(power))

def get_1x_lr_params(model):
    b = []
    b.append(model.features.deeplab.Scale.conv1)
    b.append(model.features.deeplab.Scale.bn1)
    b.append(model.features.deeplab.Scale.layer1)
    b.append(model.features.deeplab.Scale.layer2)
    b.append(model.features.deeplab.Scale.layer3)
    b.append(model.features.deeplab.Scale.layer4)
    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    b = []
    b.append(model.features.deeplab.Scale.layer5.parameters())
    for i in range(len(b)):
        for j in b[i].modules():
            for k in j.parameters():
                if k.requires_grad:
                    yield k

def get_10x_lr_guideparams(mpredict, gwindow):
    params = []
    params.append(mpredict.predict)
    #params.append(mpredict.mask_prediction.parameters())
    if gwindow is not None:
        params.append(gwindow.X_net)
        #params.append(gwindow.y_net)
        for scale_idx in range(len(gwindow.guidance_models)):
            #params.append(gwindow.guidance_models[scale_idx].appearance_net)
            params.append(gwindow.guidance_models[scale_idx].lstm) #.parameters())
    for p in params:
        for param in p.parameters():
            if param.requires_grad:
                yield param

def adjust_dimension_bound(dim_bounds, valid_bounds, target_size):
    #again, we assume it can't be larger than the total frame height
    if dim_bounds[0] < valid_bounds[0]:
        correction = dim_bounds[0] - valid_bounds[0]
        dim_bounds[0] = valid_bounds[0]
        dim_bounds[1] += correction
    if dim_bounds[1] > valid_bounds[1]:
        correction = dim_bounds[1] - valid_bounds[1]
        dim_bounds[1] = valid_bounds[1]
        dim_bounds[0] -= correction

    #expand the bounding box to fill the target dimension (if required)
    if dim_bounds[1] - dim_bounds[0] < target_size:
        correction = target_size - (dim_bounds[1] - dim_bounds[0])
        run_correction = True
        while run_correction:
            lhs_max = correction // 2
            rhs_max = correction - lhs_max
            lhs_amt = 0
            if dim_bounds[0] != valid_bounds[0]:
                lhs_amt = min(lhs_max, dim_bounds[0] - valid_bounds[0])

            rhs_amt = 0
            if dim_bounds[1] != valid_bounds[1]:
                rhs_amt = min(rhs_max, valid_bounds[1] - dim_bounds[1])

            correction = correction - (lhs_amt + rhs_amt)
            dim_bounds[0] -= lhs_amt
            dim_bounds[1] += rhs_amt
            run_correction = correction != 0

    #shrink the bounding box to fill the target dimension (if required)
    elif dim_bounds[1] - dim_bounds[0] > target_size:
        correction = (dim_bounds[1] - dim_bounds[0]) - target_size
        lhs_amt = correction // 2
        rhs_amt = correction - lhs_amt
        dim_bounds[0] += lhs_amt
        dim_bounds[1] -= rhs_amt
    return dim_bounds

def extract_instance_ROI(image, gt_mask, fwdflow, invflow, target_dims, jitter, padpx):
    img_h, img_w = image.shape[:2]
    inst_area = np.sum(gt_mask)
    #TODO: need to handle tbe case where the instance is fully occluded, and thus
    #is not present at all in the scene.
    #option #1. Have a motion model predict where the instance would be
    #option #2. Use the previous frame mask, before the instance left the scene
    #option #3. Have some no-op convention to discard the frame when there is no instance
    #option #4. Just pass in the entire scene, with a blank mask
    if inst_area == 0:
        image_roi = cv2.resize(image, (target_dims[1], target_dims[0]), interpolation=cv2.INTER_LINEAR)
        mask_roi = cv2.resize(gt_mask, (target_dims[1], target_dims[0]), interpolation=cv2.INTER_NEAREST)
        fwdflow_roi = cv2.resize(fwdflow, (target_dims[1], target_dims[0]), interpolation=cv2.INTER_LINEAR)
        invflow_roi = cv2.resize(invflow, (target_dims[1], target_dims[0]), interpolation=cv2.INTER_LINEAR)
        roi_location = [0, img_h, 0, img_w]
        return image_roi, mask_roi, fwdflow_roi, invflow_roi, roi_location

    #get the instance ROI
    rows, cols = np.where(gt_mask == 1)
    row_bounds = np.array((np.min(rows), np.max(rows)))
    col_bounds = np.array((np.min(cols), np.max(cols)))

    #TODO: for testing; artificially inflate the ROI to try to catch dropped segment components
    #eventually this will not be needed, assuming we can create a better bounding box tracker
    #also note that we only want this at testing time (I think?)
    if jitter == 0:
        row_ffactor = min(30, (row_bounds[1] - row_bounds[0])//4)
        col_ffactor = min(30, (col_bounds[1] - col_bounds[0])//4)
        row_bounds[0] = max(0, row_bounds[0] - row_ffactor)
        row_bounds[1] = min(img_h, row_bounds[1] + row_ffactor)
        col_bounds[0] = max(0, col_bounds[0] - col_ffactor)
        col_bounds[1] = min(img_w, col_bounds[1] + col_ffactor)

    obj_area_threshold = 0.75 * inst_area

    #3 options: ROI is -- {smaller, larger, mixed}
    row_smaller = (row_bounds[1] - row_bounds[0]) < target_dims[0]
    col_smaller = (col_bounds[1] - col_bounds[0]) < target_dims[1]
    rerandomize = True
    if row_smaller and col_smaller:
        #take a crop of the image and mask
        row_center = np.mean(row_bounds).astype(np.int)
        #take a crop where the object is in the scene, but at the  target dimension
        row_bounds_adj = np.array([row_center - target_dims[0] // 2, row_center + target_dims[0] // 2])
        row_bounds_adj = adjust_dimension_bound(row_bounds_adj, [0, img_h], target_dims[0])
        col_center = np.mean(col_bounds).astype(np.int)
        col_bounds_adj = np.array([col_center - target_dims[1] // 2, col_center + target_dims[1] // 2])
        col_bounds_adj = adjust_dimension_bound(col_bounds_adj, [0, img_w], target_dims[1])
        while rerandomize:
            #only jitter the bounding box to positions within bounds
            row_roi_bounds = jitter_dimension(row_bounds_adj, np.array((0, img_h)), jitter)
            col_roi_bounds = jitter_dimension(col_bounds_adj, np.array((0, img_w)), jitter)
            #re-randomize if < 50% of object in the jittered bounding box
            mask_roi = gt_mask[row_roi_bounds[0] : row_roi_bounds[1], col_roi_bounds[0] : col_roi_bounds[1]]
            obj_area = np.sum(mask_roi)
            rerandomize = obj_area < obj_area_threshold
    elif row_smaller and not col_smaller:
        #take a crop of the image and mask
        row_center = np.mean(row_bounds).astype(np.int)
        row_bounds_adj = np.array([row_center - target_dims[0] // 2, row_center + target_dims[0] // 2])
        row_bounds_adj = adjust_dimension_bound(row_bounds_adj, [0, img_h], target_dims[0])
        col_roi_bounds = np.array((max(0, col_bounds[0] - padpx), min(img_w, col_bounds[1] + padpx)))
        while rerandomize:
            row_roi_bounds = jitter_dimension(row_bounds_adj, np.array((0,img_h)), jitter)
            #re-randomize if < 50% of object in the jittered bounding box
            mask_roi = gt_mask[row_roi_bounds[0] : row_roi_bounds[1], col_roi_bounds[0] : col_roi_bounds[1]]
            obj_area = np.sum(mask_roi)
            rerandomize = obj_area < obj_area_threshold
    elif not row_smaller and col_smaller:
        #take a crop of the image and mask
        col_center = np.mean(col_bounds).astype(np.int)
        col_bounds_adj = np.array([col_center - target_dims[1] // 2, col_center + target_dims[1] // 2])
        col_bounds_adj = adjust_dimension_bound(col_bounds_adj, [0, img_w], target_dims[1])
        row_roi_bounds = np.array((max(0, row_bounds[0] - padpx), min(img_h, row_bounds[1] + padpx)))
        while rerandomize:
            col_roi_bounds = jitter_dimension(col_bounds_adj, np.array((0,img_w)), jitter)
            #re-randomize if < 50% of object in the jittered bounding box
            mask_roi = gt_mask[row_roi_bounds[0] : row_roi_bounds[1], col_roi_bounds[0] : col_roi_bounds[1]]
            obj_area = np.sum(mask_roi)
            rerandomize = obj_area < obj_area_threshold
    else:
        #add some padding to the edges
        row_roi_bounds = [max(0, row_bounds[0] - padpx), min(img_h, row_bounds[1] + padpx)]
        col_roi_bounds = [max(0, col_bounds[0] - padpx), min(img_w, col_bounds[1] + padpx)]

    #NOTE: ROI stored as [row lowbound, row highbound, col lowbound, col highbound]
    roi_location = [row_roi_bounds[0], row_roi_bounds[1], col_roi_bounds[0], col_roi_bounds[1]]
    #some sanity checks
    roi_area = (roi_location[1]-roi_location[0]) * (roi_location[3]-roi_location[2])
    assert(roi_area > 0)
    assert(row_roi_bounds[0] >= 0 and row_roi_bounds[0] < img_h)
    assert(row_roi_bounds[1] > 0 and row_roi_bounds[1] <= img_h)
    assert(col_roi_bounds[0] >= 0 and col_roi_bounds[0] < img_w)
    assert(col_roi_bounds[1] > 0 and col_roi_bounds[1] <= img_w)

    image_roi = apply_ROI_frame(image, row_roi_bounds, col_roi_bounds, target_dims)
    mask_roi = apply_ROI_frame(gt_mask, row_roi_bounds, col_roi_bounds, target_dims)
    fwdflow_roi = apply_ROI_frame(fwdflow, row_roi_bounds, col_roi_bounds, target_dims)
    invflow_roi = apply_ROI_frame(invflow, row_roi_bounds, col_roi_bounds, target_dims)
    assert(np.prod(mask_roi.shape) == np.prod(image_roi.shape[:2]))

    #TODO: this should be impossible
    if np.prod(mask_roi.shape) != np.prod(target_dims):
        print('ROI prod diff -- {} vs {}'.format(roi_location, target_dims))

    return image_roi, mask_roi, fwdflow_roi, invflow_roi, roi_location


def jitter_dimension(dim_bounds, valid_bounds, jitter):
    '''
    given an ROI, an ROI for the valid bounds, and an amount of jitter to move the
    ROI within, return a random bounding box within the valid bounds and with up to
    jitter amount of motion
    '''

    #if there is no jitter, then there's nothing to be done
    jitter = round(jitter)
    if jitter <= 0:
        return np.clip(dim_bounds, *valid_bounds)

    dim_translate = np.random.randint(-jitter, jitter)
    dim_lowbound = dim_bounds[0] + dim_translate

    correction = 0
    if dim_lowbound < valid_bounds[0]:
        correction = valid_bounds[0] - dim_lowbound
        dim_lowbound = valid_bounds[0]

    dim_highbound = dim_bounds[1] + dim_translate + correction
    if dim_highbound > valid_bounds[1]:
        correction = valid_bounds[1] - dim_highbound
        dim_highbound = valid_bounds[1]
        #NOTE: we assume here that he ROI can't be out of bounds on both the high and low bounds,
        #as that would just be crazy (and if not, we'll get it in the clipping step below)
        dim_lowbound += correction

    bbox_jitter_dim = np.array([dim_lowbound, dim_highbound])
    bbox_jitter_dim = np.clip(bbox_jitter_dim, *valid_bounds)
    return bbox_jitter_dim

def apply_ROI_frame(frame, row_roi_bounds, col_roi_bounds, target_dims=None, interp=cv2.INTER_LINEAR):
    '''
    applies the given RoI to the (numpy) array provided. Interpolates to the target_dims if:
        - target_dims is provided
        - the row or column ROI is different form the target_dims
    '''

    frame_roi = frame[row_roi_bounds[0] : row_roi_bounds[1], col_roi_bounds[0] : col_roi_bounds[1], ...]
    #resize the instance ROI to the target ROI if the dimensions don't match
    if target_dims is not None and (row_roi_bounds[1]-row_roi_bounds[0] != target_dims[0] or col_roi_bounds[1]-col_roi_bounds[0] != target_dims[1]):
        if isinstance(frame_roi, torch.Tensor):
            #TODO: I am sure there is a better way to down / upscale tensors
            frame_roi = frame_roi.numpy()
            frame_roi = cv2.resize(frame_roi, (target_dims[1], target_dims[0]), interpolation=cv2.INTER_LINEAR)
            frame_roi = torch.from_numpy(frame_roi)
        else:
            frame_roi = cv2.resize(frame_roi, (target_dims[1], target_dims[0]), interpolation=cv2.INTER_LINEAR)
    return frame_roi

def compute_gtroi(gt_mask, padpx=0):
    '''
    computes an RoI around the instance in the given gt_mask. padpx can add some #
    pixels in the X and Y dimension (to high and low bounds) of RoI
    returns a bounding box as [row low bound, row high bound, col low bound, col high bound]
    '''
    img_h, img_w = gt_mask.shape[:2]
    inst_area = np.sum(gt_mask)

    #means the object is not present in the scene; what do we do in this case?
    if inst_area == 0:
        return [-1, -1, -1, -1]

    gt_rows, gt_cols = np.nonzero(np.squeeze(gt_mask))
    row_bounds = np.clip([np.min(gt_rows) - padpx, np.max(gt_rows) + padpx], 0, img_h-1)
    col_bounds = np.clip([np.min(gt_cols) - padpx, np.max(gt_cols) + padpx], 0, img_w-1)

    #return [*row_bounds, *col_bounds]
    return [row_bounds[0], row_bounds[1], col_bounds[0], col_bounds[1]]

def flow_warp(mask, invflo):
    mask_cvt = isinstance(mask, torch.Tensor)
    if mask_cvt:
        mask = mask.data.numpy()
    mask = np.squeeze(mask)
    if isinstance(invflo, torch.Tensor):
        invflo = invflo.data.numpy()
    invflo = np.squeeze(invflo)

    height, width = mask.shape[:2]
    grid = np.array(np.dstack(np.meshgrid(np.arange(width), np.arange(height))), dtype=np.float32)
    pmap = grid + invflo
    warped_mask = cv2.remap(mask, pmap[...,0], pmap[...,1], interpolation=cv2.INTER_NEAREST)
    warped_mask = np.rint(warped_mask).astype(mask.dtype)
    if warped_mask.shape[0] != 1:
        warped_mask = np.expand_dims(warped_mask, axis=0)
    if warped_mask.shape[-1] != 1:
        warped_mask = np.expand_dims(warped_mask, axis=-1)

    if mask_cvt:
        warped_mask = torch.from_numpy(warped_mask)
    return warped_mask

def compute_flow_magnitude(fwdflow_frame, invflow_frame):
    '''
    compute the optical flow magnitude from the fwd and inv optical flow frames
    returns the normalized flow_magnitude frame (i.e. all values in [0, 1])
    '''
    flow_depth = fwdflow_frame.shape[-1]
    assert(flow_depth == invflow_frame.shape[-1])
    #NOTE: need the cast to float to broadcast to torch tensors
    for ch_idx in range(flow_depth):
        fwdmedian = float(torch.median(fwdflow_frame[...,ch_idx]))
        fwdflow_frame[...,ch_idx] = fwdflow_frame[...,ch_idx] - fwdmedian
        invmedian = float(torch.median(invflow_frame[...,ch_idx]))
        invflow_frame[...,ch_idx] = invflow_frame[...,ch_idx] - invmedian

    fwdflow_magnitude = torch.sqrt(fwdflow_frame[...,0] * fwdflow_frame[...,0] + fwdflow_frame[...,1] * fwdflow_frame[...,1])
    invflow_magnitude = torch.sqrt(invflow_frame[...,0] * invflow_frame[...,0] + invflow_frame[...,1] * invflow_frame[...,1])
    flow_magnitude = (fwdflow_magnitude + invflow_magnitude) / 2.0

    #TODO: figure out if we normalize x and y values inependently
    maxval = torch.max(flow_magnitude)
    minval = torch.min(flow_magnitude)
    flow_magnitude = (flow_magnitude - minval) / (maxval - minval)

    #add back in the 'depth' channel
    if flow_magnitude.shape[-1] != 1 and len(flow_magnitude.shape) == 3:
        flow_magnitude = torch.unsqueeze(flow_magnitude, dim=-1)

    return flow_magnitude

def remap_mask_roi(roi_mask, roi, target_dims):
    mask = np.zeros((target_dims[0], target_dims[1]), dtype=np.uint8)
    predmask_height, predmask_width = roi_mask.shape
    #reshape the output mask if it had to be downscaled
    if roi[1] - roi[0] != predmask_height or roi[3] - roi[2] != predmask_width:
        roi_mask = cv2.resize(roi_mask, (roi[3] - roi[2], roi[1] - roi[0]), interpolation=cv2.INTER_NEAREST)
    mask[roi[0]:roi[1], roi[2]:roi[3]] = roi_mask
    return mask

def generate_result_masks(segmasks, roi_locations, out_dims, mask_palette):
    '''
    out_dim --> iterable(out_height, out_width) to upscale the output to
    '''
    assert(len(segmasks) == len(roi_locations))
    #NOTE: the reason we do it this way rather than accumulate it as we go
    #is to have control over how we combine the instances, e.g. in what order
    seq_result_masks = []
    seq_result_roimasks = []

    #merge the per-instance masks
    for frame_idx, pred_frame in enumerate(segmasks):
        original_roi = roi_locations[frame_idx]
        if not is_RoI_valid(original_roi, out_dims):
            original_roi = [0, out_dims[0], 0, out_dims[1]]
        outmask = remap_mask_roi(pred_frame, original_roi, out_dims)
        pil_mask = PIL.Image.fromarray(outmask)
        if mask_palette is not None:
            pil_mask.putpalette(mask_palette)
        seq_result_masks.append(pil_mask)

        outmask_wroi = outmask.copy()
        #NOTE: ROI is stored [row_low row_high, col_low, col_high]. OpenCV orders
        #everything as (col, row) however, so I need to re-order the coordinates
        pt_A = (original_roi[2], original_roi[0])
        pt_B = (original_roi[3], original_roi[1])
        cv2.rectangle(outmask_wroi, pt_A, pt_B, 255, 2)
        outmask_wroi_pil = PIL.Image.fromarray(outmask_wroi)
        if mask_palette is not None:
            outmask_wroi_pil.putpalette(mask_palette)
        seq_result_roimasks.append(outmask_wroi_pil)
    return seq_result_masks, seq_result_roimasks

def is_RoI_valid(roi, bounds):
    '''
    roi: 4-element list (row low bound, row high bound, col low bound, col high bound)
    bounds: high bounds for valid region (0 assumed to be low bounds) [row high bound, col high bound]
    '''
    rows_valid = roi[1] > roi[0] and roi[0] >= 0 and roi[1] < bounds[0]
    cols_valid = roi[3] > roi[2] and roi[2] >= 0 and roi[3] < bounds[1]
    area_valid = ((roi[1] - roi[0]) * (roi[3] - roi[2])) > 0
    return rows_valid and cols_valid and area_valid

def save_training_outputs(mask_info, data_outpath, epochidx, seqkey, imgheight, imgwidth, mask_palette):
    '''
    saves out all the various images generated during training
    '''
    out_segmasks, roiout_segmasks = generate_result_masks(mask_info['masks'], mask_info['rois'], (imgheight, imgwidth), mask_palette)
    seq_outpath = os.path.join(data_outpath, 'final-epoch_'+str(epochidx), seqkey)
    if not os.path.exists(seq_outpath):
        os.makedirs(seq_outpath)
    save_seqoutputs_pil(out_segmasks, seq_outpath, imgheight, imgwidth)
    seq_outpath_bbox = os.path.join(data_outpath, 'bbox-final-epoch_'+str(epochidx), seqkey)
    if not os.path.exists(seq_outpath_bbox):
        os.makedirs(seq_outpath_bbox)
    save_seqoutputs_pil(roiout_segmasks, seq_outpath_bbox, imgheight, imgwidth)

    seq_outpath_grey = os.path.join(data_outpath, 'grey-final-epoch_'+str(epochidx), seqkey)
    if not os.path.exists(seq_outpath_grey):
        os.makedirs(seq_outpath_grey)

    remapped_greymasks = []
    for fidx, gmask in enumerate(mask_info['greymasks']):
        roi = mask_info['rois'][fidx]
        if not is_RoI_valid(roi, [imgheight, imgwidth]):
            roi = [0, imgheight, 0, imgwidth]
        remap_gmask = remap_mask_roi(gmask, roi, (imgheight, imgwidth))
        remapped_greymasks.append(remap_gmask)
    save_seqoutputs_greyscale(remapped_greymasks, seq_outpath_grey)

def save_seqoutputs_features(outputs, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fnames = [str(fdata['id']) for fdata in outputs]
    for frameidx, fname in enumerate(fnames):
        outpath = os.path.join(outdir, fname)
        np.savez(outpath, data=outputs[frameidx])

def compute_lossweights(mask, small_objTH=100, small_objweight=3, empty_gtweight=2):
    '''
    mask: ground truth mask (NOTE: assumed to be binary)
    small_objTH: threshold for number of pixels needed to be considered "small"
    small_objweight: weight factor for small objects
    empty_gtweight: weight factor for bg pixels on frames with no instance
    '''
    loss_weights = np.ones_like(mask)
    gt_instarea = np.sum(mask)

    #TODO: consider other cases where weighting pixel would make sense
    if gt_instarea == 0:
        loss_weights = loss_weights * empty_gtweight
    elif gt_instarea < small_objTH:
        #TODO: consider having this be more like a sigmoid than a step fcn
        loss_weights[np.nonzero(mask)] *= small_objweight

    return loss_weights

def roialign_features(X, trk_bbox, target_featdim):
    '''
    X: [batchsz, #channels, height, width]
    trk_bbox: TODO: not sure if this is [x1 y1 x2 y2]?
    target_featdim: output dimensionality to be resampled to (i.e. the guide dimension in feature space)
    '''
    trk_box = torch.Tensor(trk_bbox).unsqueeze(0)
    box_index = torch.autograd.Variable(torch.Tensor([0]))
    roi_align = RoIAlign(*target_featdim)
    X_feat = roi_align(X.cpu(), trk_box, box_index)
    return X_feat #.cuda(X.device)

