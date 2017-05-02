#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 09:55:42 2017

@author: chchen
"""

import numpy as np
import odl
import matplotlib.pyplot as plt
import scipy.ndimage.morphology as scimorph

def mean_square_error(reco,orig,mask=None):
    if mask == None:
        mask=np.ones(reco.shape,dtype=bool)
    l2_normSquared = odl.solvers.L2NormSquared(reco.space)
    reco = reco*mask
    orig = orig*mask
    norm_factor = 1.0
    diff = (reco - orig)/norm_factor
    #diff_size = np.sum(mask)
    #fom = 1.0 - (1.0/diff_size)*np.sum((diff/2)**2.0)
    fom = 1.0 - l2_normSquared(diff)/(2*(l2_normSquared(reco)+l2_normSquared(orig))) #L2-normSquared
    #fom = 1.0 - (np.sum(diff**2))/(2*np.sqrt(np.sum(reco**2))*np.sqrt(np.sum(orig**2)))
    return fom

def mean_absolute_error(reco,orig,mask=None):
    if mask is None:
        mask=np.ones(reco.shape,dtype=bool)
    l1_norm = odl.solvers.L1Norm(reco.space)
    reco = reco*mask
    orig = orig*mask
    norm_factor = 1.0
    diff = (reco - orig)/norm_factor
    fom = 1.0 - l1_norm(diff)/(l1_norm(reco) + l1_norm(orig)) #L1-norm
    return fom

def mean_density_value(reco,orig,mask=None):
    if mask is None:
        mask=np.ones(reco.shape,dtype=bool)
    reco = reco*mask
    orig = orig*mask
    reco_mean = np.sum(reco)/np.sum(mask)
    orig_mean = np.sum(orig)/np.sum(mask)
    '''
    indices = np.where(mask == True)
    reco_mean = np.mean(reco.asarray()[indices])
    orig_mean = np.mean(orig.asarray()[indices])
    '''
    fom =  1 - 0.5*(np.abs(reco_mean - orig_mean)/(np.abs(reco_mean) + np.abs(orig_mean)))
    return fom

def density_standard_deviation(reco,orig,mask=None):
    if mask is None:
        mask=np.ones(reco.shape,dtype=bool)
    reco = reco*mask
    orig = orig*mask
    # TODO: Make this continuous
    indices = np.where(mask == True)
    reco_std = np.std(reco.asarray()[indices])
    orig_std = np.std(orig.asarray()[indices])
    fom = 1 - (reco_std - orig_std)/(reco_std + orig_std)
    return fom

def density_range(reco,orig,mask=None):
    if mask is None:
        mask=np.ones(reco.shape,dtype=bool)
    indices = np.where(mask == True)
    reco_range = np.max(reco.asarray()[indices]) - np.min(reco.asarray()[indices])
    orig_range = np.max(orig.asarray()[indices]) - np.min(orig.asarray()[indices])
    fom = 1 - np.abs(reco_range - orig_range)/np.abs(reco_range + orig_range)
    return fom

def blurring(reco,orig,mask=None):
    if mask is None:
        mask=np.ones(reco.shape,dtype=bool)
    l2_normSquared = odl.solvers.L2NormSquared(reco.space)
    mask = scimorph.distance_transform_edt(1-mask)
    mask = np.exp(-mask/30)
    reco = reco*mask
    orig = orig*mask
    norm_factor = 1.0
    diff = (reco - orig)/norm_factor
    #fom = 1.0 - (np.sum(diff**2))/(2*(np.sum(reco**2)+np.sum(orig**2)))
    fom = 1.0 - l2_normSquared(diff)/(2*(l2_normSquared(reco)+l2_normSquared(orig))) #L2-normSquared
    return fom

def false_structures(reco,orig,mask=None):
    if mask is None:
        mask=np.ones(reco.shape,dtype=bool)
    l2_normSquared = odl.solvers.L2NormSquared(reco.space)
    mask = scimorph.distance_transform_edt(1-mask)
    mask = np.exp(-mask/30)
    if len(np.unique(mask)) != 1:
        mask = 1-mask
    reco = reco*mask
    orig = orig*mask
    norm_factor = 1.0
    diff = (reco - orig)/norm_factor
    fom = 1.0 - l2_normSquared(diff)/(2*(l2_normSquared(reco)+l2_normSquared(orig))) #L2-normSquared
    return fom

'''
# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[100, 100], dtype='float32')

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 360)
# Detector: uniformly sampled, n = 558, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 558)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform (= forward projection). We use the 'skimage' backend.
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

mse = []
mae = []
blur = []
false_struct = []

mask = (np.asarray(phantom) == 1)

for i in np.linspace(0.1,1,10):
    phantom_noisy = phantom + odl.phantom.white_noise(reco_space, stddev=i)
    mse.append(mean_square_error(phantom_noisy, phantom))
    mae.append(mean_absolute_error(phantom_noisy, phantom))
    blur.append(blurring(phantom_noisy, phantom, mask))
    false_struct.append(false_structures(phantom_noisy, phantom, mask))
plt.plot(mse)
#plt.plot(mae)


# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(phantom)

# Back-projection can be done by simply calling the adjoint operator on the
# projection data (or any element in the projection space).
backproj = ray_trafo.adjoint(proj_data)

# Shows a slice of the phantom, projections, and reconstruction
phantom.show(title='Phantom')
proj_data.show(title='Projection data (sinogram)')
backproj.show(title='Back-projected data')
'''
