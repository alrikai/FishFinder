import sys

import numpy as np
import numpy.linalg as linalg
import cv2

import scipy
import scipy.ndimage

def is_invertible(M):
    non_singular =  M.shape[0] == M.shape[1] and np.linalg.matrix_rank(M) == M.shape[0]
    stable = linalg.cond(M) < 1/sys.float_info.epsilon
    return non_singular and stable

#NOTE: img_center is (height, width)
#translate is (col_offset, row_offset)
#the returned M is in [x; y; 1] format
def compute_affine_T(rotate, scale, translate, center):
    rad_rot = np.deg2rad(rotate)
    T = np.array([[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]])
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    R = np.array([[np.cos(rad_rot), -np.sin(rad_rot), 0], [np.sin(rad_rot), np.cos(rad_rot), 0], [0, 0, 1]])
    Tc_to = np.array([[1, 0, -center[1]], [0, 1, -center[0]], [0, 0, 1]])
    Tc_from = np.array([[1, 0, center[1]], [0, 1, center[0]], [0, 0, 1]])
    #move center coord to origin, rotate
    M = np.matmul(R, Tc_to)
    #scale up
    M = np.matmul(S, M)
    #move scaled, rotated back to center coord
    M = np.matmul(Tc_from, M)
    #translate final object
    M = np.matmul(T, M)
    if not is_invertible(M):
        print ("NOTE: transformation matrix not invertible, headed for trouble...")
    return M

def get_coords(img_height, img_width, rcorder, homogeneous=False):
    rcoord = np.repeat(np.arange(img_height), img_width)
    ccoord = np.tile(np.arange(img_width), img_height)
    if homogeneous:
        hcoord = np.ones_like(rcoord)
    if rcorder == 'rc':
        if homogeneous:
            coords = list(zip(rcoord, ccoord, hcoord))
        else:
            coords = list(zip(rcoord, ccoord))
    else:
        if homogeneous:
            coords = list(zip(ccoord, rcoord, hcoord))
        else:
            coords = list(zip(ccoord, rcoord))
    return coords

def warp_coordinates_T(M, img_height, img_width):
    coords_h = get_coords(img_height, img_width, 'cr', homogeneous=True)
    xform_coords_h = np.dot(coords_h, M)
    xform_coords = np.fliplr(xform_coords_h[..., :2])
    warp_coords = xform_coords.reshape(img_height, img_width, 2)
    return warp_coords

def affinewarp_image_T(image, M, interpol=cv2.INTER_CUBIC):
    img_height, img_width = image.shape[:2]
    warp_coords = warp_coordinates_T(M, img_height, img_width)
    res_image = cv2.remap(image, warp_coords[...,1].astype(np.float32), warp_coords[...,0].astype(np.float32), interpol)
    return res_image

def warp_coordinates_T_wcoords(M, coords_h, img_height, img_width):
    xform_coords_h = np.dot(coords_h, M)
    xform_coords = np.fliplr(xform_coords_h[..., :2])
    warp_coords = xform_coords.reshape(img_height, img_width, 2)
    return warp_coords

def affinewarp_image_T_wcoords(image, M, coords_h, interpol=cv2.INTER_CUBIC):
    img_height, img_width = image.shape[:2]
    warp_coords = warp_coordinates_T_wcoords(M, coords_h, img_height, img_width)
    res_image = cv2.remap(image, warp_coords[...,1].astype(np.float32), warp_coords[...,0].astype(np.float32), interpol)
    return res_image

def elastic_deformation(data, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = data.shape
    deform_shape = shape[0:2]
    #compute the data grid deformations
    dx = scipy.ndimage.filters.gaussian_filter((random_state.rand(*deform_shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
    dy = scipy.ndimage.filters.gaussian_filter((random_state.rand(*deform_shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
    #get the un-deformed grid
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    #compute the deformed grid
    xy_indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    elastic_mask = scipy.ndimage.interpolation.map_coordinates(data, xy_indices, order=1).reshape(deform_shape)
    return elastic_mask
