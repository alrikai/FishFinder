import cv2
import numpy as np

import utils.transform_utils as xformutils

class MaskAugmenter():
    def __init__(self, r_angle=5, s_range=0.1, t_amt=0.05):

        #+- 5% rotation (around the center of the object)
        self.rotate_range = (-r_angle, r_angle)
        #+- 10% scaling up / down of the object
        self.scale_range = (1-s_range, 1+s_range)
        self.translate_factor = t_amt
        self.T_min = 3
        self.T_max = 30

        self.elastic_alpha = 30
        self.elastic_sigma = 5
        self.kernel_dims = (5,5)
        self.morph_kernel = np.ones(self.kernel_dims, np.uint8)

        self.imh = None
        self.imw = None
        self.base_coords = None

    def jitter_mask(self, inmask):
        '''
        coarsens / offsets the given (uint8) image; useful for coarsening the gt mask
        to use it as the previous mask of the next frame during training
        '''
        assert(inmask.dtype == np.uint8)
        #if an empty mask (i.e. complete occlusion, etc) then dont do anything
        if np.sum(inmask) == 0:
            return inmask

        mask = np.squeeze(inmask)
        imheight, imwidth = mask.shape
        if self.imh != imheight or self.imh != imwidth or self.base_coords is None:
            self.imh = imheight
            self.imw = imwidth
            self.base_coords = xformutils.get_coords(self.imh, self.imw, 'cr', homogeneous=True)

        #run dilation, to blur the finer mask details
        mask = cv2.dilate(mask, self.morph_kernel, iterations=1)

        #compute the allowable translation bounds based on the instance size
        inst_row, inst_col = np.nonzero(mask)
        inst_sizes = [np.max(inst_row) - np.min(inst_row), np.max(inst_col) - np.min(inst_col)]
        #put min and max bounds, otherwise small objects wont move at all,
        #and large objects will move too much
        translate_bounds = [min(self.T_max, max(self.T_min, self.translate_factor * dim)) for dim in inst_sizes]
        row_mean = np.mean(inst_row)
        col_mean = np.mean(inst_col)
        tmask = self.safe_affine_transform(mask, (row_mean, col_mean), translate_bounds)

        #run the elastic (non-rigid) deformation
        emask = xformutils.elastic_deformation(tmask, self.elastic_alpha, self.elastic_sigma)

        #NOTE: enforce that the output frame is the same dimension as the input
        #--> we can sort of cheat, since we know the only dimension differences
        #will be from the batchsz or the #channels (both of which are just 1)
        if emask.shape[0] != inmask.shape[0]:
            emask = np.expand_dims(emask, axis=0)
        if emask.shape[-1] != inmask.shape[-1]:
            emask = np.expand_dims(emask, axis=-1)
        return emask.astype(np.uint8)

    #NOTE: this assumes obj_data is the original data
    def safe_affine_transform(self, obj_mask, inst_center, translate_bounds, interpol_method=cv2.INTER_NEAREST):
        rotate_angle = np.random.uniform(*self.rotate_range)
        scale_factor = np.random.uniform(*self.scale_range)
        #randomly choose to move up or down, and left or right.
        translate = np.random.uniform(-1, 1, 2) * translate_bounds
        #translate = np.rint(T_shift).astype(np.int32)
        xform_M = xformutils.compute_affine_T(rotate_angle, scale_factor, translate, inst_center)
        base_coords_h = np.copy(self.base_coords)
        xform_image = xformutils.affinewarp_image_T_wcoords(obj_mask, xform_M, base_coords_h, interpol=interpol_method)
        return xform_image
