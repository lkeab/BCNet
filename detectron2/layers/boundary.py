#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 00:22:15 2020

@author: fanq15
"""
import numpy as np
#from PIL import Image #, ImageOps, ImageDraw
from skimage import filters, img_as_ubyte
from skimage.morphology import remove_small_objects, dilation, erosion, binary_dilation, binary_erosion, square
#from scipy.ndimage.interpolation import map_coordinates
#from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import center_of_mass

def get_contour_interior(mask, bold=False):
    if True: #'camunet' == config['param']['model']:
        # 2-pixel contour (1out+1in), 2-pixel shrinked interior
        outer = binary_dilation(mask) #, square(9))
        if bold:
            outer = binary_dilation(outer) #, square(9))
        inner = binary_erosion(mask) #, square(9))
        contour = ((outer != inner) > 0).astype(np.uint8)
        interior = (erosion(inner) > 0).astype(np.uint8)
    else:
        contour = filters.scharr(mask)
        scharr_threshold = np.amax(abs(contour)) / 2.
        contour = (np.abs(contour) > scharr_threshold).astype(np.uint8)
        interior = (mask - contour > 0).astype(np.uint8)
    return contour, interior

def get_center(mask):
    r = 2
    y, x = center_of_mass(mask)
    center_img = Image.fromarray(np.zeros_like(mask).astype(np.uint8))
    if not np.isnan(x) and not np.isnan(y):
        draw = ImageDraw.Draw(center_img)
        draw.ellipse([x-r, y-r, x+r, y+r], fill='White')
    center = np.asarray(center_img)
    return center

def get_instances_contour_interior(instances_mask):
    adjacent_boundary_only = False #False #config['contour'].getboolean('adjacent_boundary_only')
    instances_mask = instances_mask.data
    result_c = np.zeros_like(instances_mask, dtype=np.uint8)
    result_i = np.zeros_like(instances_mask, dtype=np.uint8)
    weight = np.ones_like(instances_mask, dtype=np.float32)
    #masks = decompose_mask(instances_mask)
    #for m in masks:
    contour, interior = get_contour_interior(instances_mask, bold=adjacent_boundary_only)
    #center = get_center(m)
    if adjacent_boundary_only:
        result_c += contour
    else:
        result_c = np.maximum(result_c, contour)
    result_i = np.maximum(result_i, interior)
    #contour += center
    contour = np.where(contour > 0, 1, 0)
    # magic number 50 make weight distributed to [1, 5) roughly
    weight *= (1 + gaussian_filter(contour, sigma=1) / 50)
    if adjacent_boundary_only:
        result_c = (result_c > 1).astype(np.uint8)
    return result_c, result_i, weight
