"""
Bregman-TV reconstruction example for simulated Skull CT data
"""
import odl
import numpy as np
import os
import adutils
import sys
import time

import psutil
from pympler.tracker import SummaryTracker

# ----------------------------------------------------------------------------
# Create a text file and write everything both in terminal and in file
# ----------------------------------------------------------------------------
savePath = '/home/aringh/Documents/'

if not os.path.exists(savePath):
    os.makedirs(savePath)

time_now = time.strftime("%Y_%m_%d__%H_%M_%S")
output_filename = 'Terminal_output_' + time_now + '.txt'


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


# ----------------------------------------------------------------------------
# Set up tomography problem
# ----------------------------------------------------------------------------
# Discretization
space = adutils.get_discretization()

# Forward operator (in the form of a broadcast operator)
A = adutils.get_ray_trafo(space, use_subset=True, use_rebin=False, rebin_factor=100)

# Data
rhs = adutils.get_data(A, use_subset=True, use_rebin=False, rebin_factor=100)


# ----------------------------------------------------------------------------
# Set up the optimization problem
# ----------------------------------------------------------------------------
# Construct operators and functionals
gradient = odl.Gradient(space)

# Column vector of operators
op = odl.BroadcastOperator(A, gradient)

Anorm = odl.power_method_opnorm(A,
                                xstart=odl.phantom.white_noise(A.domain),
                                maxiter=2)
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


# ----------------------------------------------------------------------------
# Helper functions to print things in each iteration
# ----------------------------------------------------------------------------
# Callback for the solver that print CPU/RAM use
class CallbackMyCallback(odl.solvers.util.callback.SolverCallback):

    """Callback for printing memory and CPU usage."""

    def __init__(self, step=1):
        """Initialize a new instance.

        Parameters
        ----------
        step : positive int, optional
            Number of iterations between output. Default: 1

        Examples
        --------
        Print memory and CPU usage

        >>> callback = CallbackPrintMemory()
        """
        self.step = step
        self.iter = 0

    def __call__(self, _):
        """Print the memory and CPU usage"""
        if self.iter % self.step == 0:
            print('CPU usage (% each core): {}'.format(
                  psutil.cpu_percent(percpu=True)))
            print('RAM usage: {}'.format(psutil.virtual_memory()))
            print('SWAP usage: {}'.format(psutil.swap_memory()))

        self.iter += 1

    def reset(self):
        """Set `iter` to 0."""
        self.iter = 0

    def __repr__(self):
        """Return ``repr(self)``."""
        optargs = [('step', self.step, 1)]
        inner_str = odl.util.signature_string([], optargs)
        return '{}({})'.format(self.__class__.__name__, inner_str)


# Callback for the solver that print objects in memory
class CallbackMySecondCallback(odl.solvers.util.callback.SolverCallback):

    """Callback for printing objects in python memory."""

    def __init__(self, step=1):
        """Initialize a new instance.

        Parameters
        ----------
        step : positive int, optional
            Number of iterations between output. Default: 1

        Examples
        --------
        Print memory and CPU usage

        >>> callback = CallbackPrintMemory()
        """
        self.step = step
        self.iter = 0
        self.tracker = SummaryTracker()

    def __call__(self, _):
        """Print the memory and CPU usage"""
        if self.iter % self.step == 0:
            self.tracker.print_diff()

        self.iter += 1

    def reset(self):
        """Set `iter` to 0."""
        self.iter = 0

    def __repr__(self):
        """Return ``repr(self)``."""
        optargs = [('step', self.step, 1)]
        inner_str = odl.util.signature_string([], optargs)
        return '{}({})'.format(self.__class__.__name__, inner_str)


callback = (odl.solvers.CallbackPrintIteration() &
            CallbackMyCallback() &
            CallbackMySecondCallback())

#callback = odl.solvers.CallbackPrintIteration()

# ----------------------------------------------------------------------------
# Solve the optimization problem
# ----------------------------------------------------------------------------
# Run such that last iteration is saved (saveReco = 1) or none (saveReco = 0)
niter = 1000
odl.solvers.chambolle_pock_solver(x, f, g, op, tau=tau, sigma=sigma,
                                  niter=niter, gamma=gamma, callback=callback)

# Close the log-file and set standard output to terminal
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
