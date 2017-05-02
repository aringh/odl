# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import odl
import numpy as np
import eval_of_fom
import fom_test_shepp_logan

# Seed the randomness
np.random.seed(1)

# Discrete reconstruction space.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300], dtype='float32')

# Make a parallel beam geometry with flat detector
angle_partition = odl.uniform_partition(0, np.pi, 360)
detector_partition = odl.uniform_partition(-30, 30, 558)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Create the forward operator
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)


# Create phantom
discr_phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Create sinogram of forward projected phantom with noise
data_noise_free = ray_trafo(discr_phantom)

noise_list = [0.01, 0.03, 0.1, 2]
data_list = [None] * len(noise_list)

for noise_level, i in zip(noise_list, range(len(data_list))):
    noise = odl.phantom.white_noise(ray_trafo.range)
    noise = noise * data_noise_free.norm()/noise.norm()
    data_list[i] = data_noise_free + noise_level * noise


"""
# Some temporary FOM's.
class L2fom(object):
    def __call__(self, reco, phantom):
        l2_func = odl.solvers.L2Norm(reco.space)
        return l2_func(reco-phantom)/l2_func(phantom)


class L1fom(object):
    def __call__(self, reco, phantom):
        l1_func = odl.solvers.L1Norm(reco.space)
        return l1_func(reco-phantom)/l1_func(phantom)


class stdfom(object):
    def __call__(self, reco, phantom):
        return np.abs(np.std(reco)-np.std(phantom))/np.std(phantom)

fom_list = [L2fom(), L1fom(), stdfom()]
"""

# Create some reconstruction operators
fbp_op_1 = odl.tomo.fbp_op(ray_trafo, padding=True, filter_type='Ram-Lak',
                           frequency_scaling=0.1)

fbp_op_2 = odl.tomo.fbp_op(ray_trafo, padding=True, filter_type='Cosine',
                           frequency_scaling=0.1)

fbp_op_3 = odl.tomo.fbp_op(ray_trafo, padding=True, filter_type='Hamming',
                           frequency_scaling=0.1)

fbp_op_4 = odl.tomo.fbp_op(ray_trafo, padding=True, filter_type='Ram-Lak',
                           frequency_scaling=0.3)

fbp_op_5 = odl.tomo.fbp_op(ray_trafo, padding=True, filter_type='Cosine',
                           frequency_scaling=0.3)

fbp_op_6 = odl.tomo.fbp_op(ray_trafo, padding=True, filter_type='Hamming',
                           frequency_scaling=0.3)

reco_op_list = [fbp_op_1, fbp_op_2, fbp_op_3, fbp_op_4, fbp_op_5, fbp_op_6]


# Create a list of FOMs
fom_list = [fom_test_shepp_logan.mean_square_error,
            fom_test_shepp_logan.mean_absolute_error,
            fom_test_shepp_logan.density_range,
            fom_test_shepp_logan.false_structures,
            fom_test_shepp_logan.blurring]

output = eval_of_fom.fom_eval(fom_list, fbp_op_1, data_list, discr_phantom)

# eval_of_fom.mean_confidence_interval(output[0,:]-output[1,:]))

test_mat = eval_of_fom.confidence_interval_t_dist(output, conf_level=0.95,
                                                  axis=1)

d_mat = eval_of_fom.compare_reco_matrix(fom_list, reco_op_list, data_list,
                                        discr_phantom, conf_level=0.5)

print(np.round(d_mat, decimals=3))
