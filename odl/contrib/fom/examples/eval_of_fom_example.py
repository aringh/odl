"""Simple test of comparison of reconstruction operators using FOMs.

Numerical test of a few fbp reconstruction operators (with different filtering)
using some FOMS and data with diffrent noise levels.
"""

import odl
import odl.contrib.fom as fom
import numpy as np

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

for i, noise_level in enumerate(noise_list):
    noise = odl.phantom.white_noise(ray_trafo.range)
    noise = noise * data_noise_free.norm()/noise.norm()
    data_list[i] = data_noise_free + noise_level * noise

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
fom_list = [fom.mean_squared_error,
            fom.mean_absolute_error,
            fom.range_difference,
            fom.ssim,
            fom.blurring]

d_mat = fom.compare_reco_matrix(fom_list, reco_op_list, data_list,
                                discr_phantom, conf_level=0.5)

print(np.round(d_mat, decimals=3))
