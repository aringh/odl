#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 09:17:23 2017

@author: aringh
"""

import odl
import numpy as np

# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300], dtype='float32')

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 360)
# Detector: uniformly sampled, n = 558, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 558)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# The implementation of the ray transform to use, options:
# 'scikit'                    Requires scikit-image (can be installed by
#                             running ``pip install scikit-image``).
# 'astra_cpu', 'astra_cuda'   Require astra tomography to be installed.
#                             Astra is much faster than scikit. Webpage:
#                             https://github.com/astra-toolbox/astra-toolbox
impl = 'astra_cuda'

# Create the forward operator
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl=impl)


# Create phantom
discr_phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Create sinogram of forward projected phantom with noise
noise_free_data = ray_trafo(discr_phantom)

noise = odl.phantom.white_noise(ray_trafo.range)
noise = noise * 1/noise.norm() * noise_free_data.norm() * 0.10

data = noise_free_data + noise


mu = 0.1

# TV-term
gradient = odl.Gradient(reco_space)
g_TV = mu * odl.solvers.L1Norm(gradient.range)


# Data discrepency, 2-norm
g_2norm = 0.5 * odl.solvers.L2NormSquared(ray_trafo.range).translated(data)

g = [g_TV, g_2norm]
lin_ops = [gradient, ray_trafo]

grad_norm = odl.power_method_opnorm(gradient)
l2_norm = odl.power_method_opnorm(ray_trafo)

# Create a zero-functional
f = odl.solvers.ZeroFunctional(reco_space)

x_tv = reco_space.zero()
callback_tv = odl.solvers.CallbackPrintIteration()
odl.solvers.forward_backward_pd(x_tv, odl.solvers.ZeroFunctional(reco_space),
                                g, lin_ops, odl.solvers.ZeroFunctional(reco_space),
                                tau=0.4, sigma=[1/grad_norm**2, 1/l2_norm**2],
                                niter=2000, callback=callback_tv)

x_tv.show('TV-reconstruction')

#class BregmanOneHomo(odl.solvers.Functional):
#
#    """..."""
#
#    def __init__(self, f, p):
#        """Initialize a new instance.
#
#        Parameters
#        ----------
#        """
#        self.__f = f
#        self.__p = p
#        self.__to_eval = f - odl.solvers.QuadraticForm(vector=p)
#        super().__init__(space=self.__f.domain, linear=False,
#                         grad_lipschitz=np.nan)
#
#    # TODO: update when integration operator is in place: issue #440
#    def _call(self, x):
#        self.__to_eval(x)
#
#    @property
#    def convex_conj(self):
#        """..."""
#        return self.__f.convex_conj.translated(-p)


# Create initial guess for the solver.
x = reco_space.zero()
p = reco_space.zero()  # Initial value of the subgradient

callback_out = odl.solvers.CallbackShow(display_step=1,
                                        saveto='/home/aringh/Documents/bregman_iter{}.png')

for breg_iter in range(10):
    print('Outer Bregman iteration: {}'.format(breg_iter))

    h = mu * odl.solvers.QuadraticForm(vector=-p)

    # Used to display intermediate results and print iteration number.
    callback_inner = odl.solvers.CallbackPrintIteration()

    # Call the solver. x is updated in-place with the consecutive iterates.
    odl.solvers.forward_backward_pd(x, f, g, lin_ops, h, tau=0.4,
                                    sigma=[1/grad_norm**2, 1/l2_norm**2],
                                    niter=2000, callback=callback_inner)

    callback_out(x)

    p -= 1/mu * ray_trafo.adjoint(ray_trafo(x) - data)



# Display images
discr_phantom.show(title='Phantom')
data.show(title='Simulated data (Sinogram)')
x.show(title='Bregman-TV', force_show=True)