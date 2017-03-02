"""
TV reconstruction example for simulated Skull CT data
"""

import odl
import numpy as np
import os
import time
import adutils
import sys


rebin_factor = 10

# This should be run only once.
# adutils.rebin_data(rebin_factor)

# Discretization
reco_space = adutils.get_discretization()

# Forward operator (in the form of a broadcast operator)
A = adutils.get_ray_trafo(reco_space, use_rebin=True,
                          rebin_factor=rebin_factor)

# Data
rhs = adutils.get_data(A, use_rebin=True, rebin_factor=rebin_factor)


# Run such that last iteration is saved (saveReco = 1) or none (saveReco = 0)
saveReco = True
savePath = '/home/aringh/Documents/Reco_Simulated_120kV/'
#savePath = '/home/aringh/nas_data/aringh/Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy/Bregman/'

time_now = time.strftime("%Y_%m_%d__%H_%M_%S")
folder = ('Param_search_Bregman_downsample_' + str(rebin_factor) + '__' +
          time_now)

directory = savePath + folder

if saveReco and not os.path.exists(directory):
    os.makedirs(directory)


output_filename = 'Terminal_output.txt'

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

sys.stdout = Logger(directory + '/' +output_filename)



niter = 2000
nbreg_iter = 10

# Gradient operator
gradient = odl.Gradient(reco_space)

# Identity operator
id_op = odl.IdentityOperator(reco_space)

# Column vector of operators
op = odl.BroadcastOperator(A, gradient, id_op)

Anorm = odl.power_method_opnorm(A[1], maxiter=2)
Dnorm = odl.power_method_opnorm(gradient,
                                xstart=odl.phantom.white_noise(gradient.domain),
                                maxiter=10)

# Estimated operator norm, add 10 percent (+ 1.0 is for id_op)
op_norm = 1.1 * np.sqrt(len(A.operators)*(Anorm**2) + Dnorm**2 + 1.0)

print('Norm of the product space operator: {}'.format(op_norm))

# l2-squared data matching
l2_norm = odl.solvers.L2NormSquared(A.range).translated(rhs)

# Set g functional to zero
g = odl.solvers.ZeroFunctional(op.domain)

# Create a zero-functional
f = odl.solvers.IndicatorBox(reco_space, lower=0.0)

# Use initial guess
x_init = adutils.get_initial_guess(reco_space)


# Acceleration parameter
gamma = 0.4

# Step size for the proximal operator for the primal variable x
tau = 1.0 / op_norm

# Step size for the proximal operator for the dual variable y
sigma = 1.0 / op_norm  # 1.0 / (op_norm ** 2 * tau)

#lambs = (0.05, 0.03, 0.01, 0.005, 0.003, 0.001)

lambs = (0.1, 0.05, 0.01, 0.005)

for lamb in lambs:
    print('Regularization paramenter: {}'.format(lamb))
    # Isotropic TV-regularization i.e. the l1-norm
    #l1_norm = lamb * odl.solvers.L1Norm(gradient.range)
    l1_norm = lamb * odl.solvers.GroupL1Norm(gradient.range)

    # Reconstruct
#    callbackShowReco = (odl.solvers.CallbackPrintIteration() &  # Print iter
#                        odl.solvers.CallbackShow(coords=[None, 0, None],
#                                                 display_step=20,
#                                                 clim=[0.018, 0.022]) &  # Show
#                        odl.solvers.CallbackShow(coords=[0, None, None],
#                                                 display_step=20,
#                                                 clim=[0.018, 0.022]) &
#                        odl.solvers.CallbackShow(coords=[None, None, 60],
#                                                 display_step=20,
#                                                 clim=[0.018, 0.022]))

    callbackPrintIter = odl.solvers.CallbackPrintIteration()

    # Use the FBP as initial guess
    x = x_init.copy()  # =  reco_space.zero()
    p = (l1_norm*gradient).gradient(x)  # reco_space.zero()

    for breg_iter in range(nbreg_iter):
        print('Outer Bregman iteration: {}'.format(breg_iter))
        # The linear term for the one-homogenious case
        bregman_term = lamb * odl.solvers.QuadraticForm(vector=-p)
        # This instead makes it into (almost) ordinary TV
        #bregman_term = odl.solvers.ZeroFunctional(reco_space)

        # Combine functionals
        f = odl.solvers.SeparableSum(l2_norm, l1_norm, bregman_term)

        # Used to display intermediate results and print iteration number.
        callback_inner = odl.solvers.CallbackPrintIteration()

        odl.solvers.chambolle_pock_solver(x, f, g, op, tau=tau, sigma=sigma,
                                          niter=niter, gamma=gamma,
                                          callback=callbackPrintIter)

        # callbackShowReco.reset()
        #p -= 1/lamb * A.adjoint(A(x) - rhs)
        p -= 1/lamb * A.adjoint((l2_norm.gradient)(A(x)))

        # This does not return the same value as the implementation above. Are
        # there more than one subdifferential in the point?
        #p = ((g_TV*gradient).gradient)(x)


        if saveReco:
            saveName = ('TV-Bregman__lambda_' +
                        str(lamb).replace('.', 'point') + '__Bregman_iter_' +
                        str(breg_iter))
            #x.show(title=('Lambda {}, Bregman iteration {}, slice 1'
            #              ''.format(lamb, breg_iter)),
            #       coords=[None, 0, None],
            #       clim=[0.018, 0.022],
            #       saveto=directory+'/'+saveName+'__slice_1')
            #x.show(title=('Lambda {}, Bregman iteration {}, slice 2'
            #              ''.format(lamb, breg_iter)),
            #       coords=[0, None, None],
            #       clim=[0.018, 0.022],
            #       saveto=directory+'/'+saveName+'__slice_2')
            #x.show(title=('Lambda {}, Bregman iteration {}, slice 3'
            #              ''.format(lamb, breg_iter)),
            #       coords=[None, None, 60],
            #       clim=[0.018, 0.022],
            #       saveto=directory+'/'+saveName+'__slice_3')
            np.save(directory+'/'+saveName, np.asarray(x))

            #x.show(title=('Lambda {}, Bregman iteration {}, slice 1'
            #              ''.format(lamb, breg_iter)),
            #       coords=[None, 0, None],
            #       saveto=directory+'/'+saveName+'__slice_1_1')
            #x.show(title=('Lambda {}, Bregman iteration {}, slice 2'
            #              ''.format(lamb, breg_iter)),
            #       coords=[0, None, None],
            #       saveto=directory+'/'+saveName+'__slice_2_1')
            #x.show(title=('Lambda {}, Bregman iteration {}, slice 3'
            #              ''.format(lamb, breg_iter)),
            #       coords=[None, None, 60],
            #       saveto=directory+'/'+saveName+'__slice_3_1')
            np.save(directory+'/'+saveName, np.asarray(x))

# # This dumps the reconstruction and other things to disc
# import pickle
# import time
# time_now = time.strftime("%Y_%m_%d__%H_%M_%S")
#
# with open('recon_' + time_now + '.pickle', 'wb') as f:
#     pickle.dump([reco_space, A, rhs, x], f)

sys.stdout.log.close
sys.stdout = sys.stdout.terminal
