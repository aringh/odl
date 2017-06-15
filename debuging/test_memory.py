"""
Bregman-TV reconstruction example for simulated Skull CT data
"""
import odl
import numpy as np
import os
import adutils
import sys
import psutil
import time

saveReco = True
savePath = '/home/aringh/Documents/'


# Create a text file and write everything both in terminal and in file
if not os.path.exists(savePath):
    os.makedirs(savePath)

time_now = time.strftime("%Y_%m_%d__%H_%M_%S")
output_filename = 'Terminal_output_' + time_now + '.txt'
#output_filename = 'Terminal_output_TV.txt'

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(savePath + output_filename)


# Discretization
space = adutils.get_discretization()

# Forward operator (in the form of a broadcast operator)
A = adutils.get_ray_trafo(space)

# Data
rhs = adutils.get_data(A)


## FBP-recon
#fbp_op = adutils.get_fbp(A)
#x_fbp = fbp_op(rhs)
#
#if saveReco:
#    saveName = savePath + 'Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_FBP.npy'
#    np.save(saveName, np.asarray(x_fbp))
#
#print('FBP reconstruction is done!')

# Bregman-TV recon

# Construct operators and functionals
gradient = odl.Gradient(space)

# Column vector of operators
op = odl.BroadcastOperator(A, gradient)

Anorm = odl.power_method_opnorm(A, maxiter=2)
Dnorm = odl.power_method_opnorm(gradient,
                                xstart=odl.phantom.white_noise(gradient.domain),
                                maxiter=10)

# Estimated operator norm, add 10 percent
op_norm = 1.1 * np.sqrt(len(A.operators)*(Anorm**2) + Dnorm**2)

print('Norm of the product space operator: {}'.format(op_norm))

lamb = 0.005  # l2NormGrad/l1NormGrad = 0.01

# l2-squared data matching
l2_norm = odl.solvers.L2NormSquared(A.range).translated(rhs)

# Isotropic TV-regularization i.e. the l1-norm
l1_norm = lamb * odl.solvers.L1Norm(gradient.range)

# Combine functionals
f = odl.solvers.SeparableSum(l2_norm, l1_norm)

# Set g functional to zero
g = odl.solvers.ZeroFunctional(op.domain)

# Accelerataion parameter
gamma = 0.4

# Step size for the proximal operator for the primal variable x
tau = 1.0 / op_norm

# Step size for the proximal operator for the dual variable y
sigma = 1.0 / op_norm  # 1.0 / (op_norm ** 2 * tau)

# Use initial guess
x = A.domain.zero()

# Callback for the solver that both print iteration number and CPU/RAM use
class CallbackMycallback(odl.solvers.util.callback.SolverCallback):

    def __init__(self):
        self.print_iter_callback = odl.solvers.CallbackPrintIteration()

    def __call__(self, x):
        self.print_iter_callback(x)
        print('CPU usage: {}'.format(psutil.cpu_percent(percpu=True)))
        print('RAM usage: {}'.format(psutil.virtual_memory()))
        print('SWAP usage: {}'.format(psutil.swap_memory()))

    def reset(self):
        self.print_iter_callback.reset()

callback_print_iter = CallbackMycallback()

## Reconstruct
#callbackShowReco = (odl.solvers.CallbackPrintIteration() &  # Print iterations
#                    odl.solvers.CallbackShow(coords=[None, 0, None]) &  # Show parital reconstructions
#                    odl.solvers.CallbackShow(coords=[0, None, None]) &
#                    odl.solvers.CallbackShow(coords=[None, None, 60]))

#callbackPrintIter = odl.solvers.CallbackPrintIteration()

# Run such that last iteration is saved (saveReco = 1) or none (saveReco = 0)
saveReco = False
savePath = '/home/user/Simulated/120kV/'
niter = 1000
odl.solvers.chambolle_pock_solver(x, f, g, op, tau=tau, sigma = sigma,
				  niter = niter, gamma=gamma, callback=callback_print_iter)

# Close the log-file and set standard output to terminal
sys.stdout.log.close
sys.stdout = sys.stdout.terminal
