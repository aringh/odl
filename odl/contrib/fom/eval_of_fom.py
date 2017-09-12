# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import scipy.stats

__all__ = ('fom_eval', 'confidence_interval_t_dist', 'compare_reco_matrix')

def fom_eval(fom_list, reco_op, data_list, phantom):
    """Evaluation of figure of merits (FOM).

    Evaluates a set of FOM's for a reconstruction mehtod on a set of data.

    Parameters
    ----------
    fom_list : list
        List of callable where each element corresponds to one FOM.
    reco_op : callable
        Reconstruction operator, takes data as input and outputs a
        reconstruction.
    data_list : list
        List of data sets.
    phantom : `FnBase` or `ProductSpace`
        The phantom with respect to which the FOM's are computed.

    Returns
    -------
    fom_vals : numpy array
        Matrix containing evaluation of the FOM's. Each row corresponds to a
        FOM and each column corresponds to a data set.
    """
    num_fom = len(fom_list)
    num_data = len(data_list)
    fom_vals = np.zeros([num_fom, num_data])

    for fom, i in zip(fom_list, range(num_fom)):
        for data, j in zip(data_list, range(num_data)):
            reco = reco_op(data)
            fom_vals[i, j] = fom(reco, phantom)

    return fom_vals


def confidence_interval_t_dist(data, conf_level=0.95, axis=1):
    """Computes a confidence interval around the data.

    This computes a confidence interval around the mean of the data, based on
    Student's t-distribution.

    Parameters
    ----------
    data : list
        Maxtrix of data. One confidence interval is computed for each set of
        values along the dimension ``axis``.
    conf_level : float, optional
        Value that defines the confidnece level.
    axis : int, optional
        Axis along which the confidence levels are computed.

    Returns
    -------
    m, m-h, m+h : numpy arrays
        m is the mean values, and m-h and m+h defines the confidence interval
        [m-h, m+h].
    """
    n = data.shape[1]
    m, se = np.mean(data, axis), scipy.stats.sem(data, axis)
    h = se * scipy.stats.t._ppf((1+conf_level)/2., n-1)
    return m, m-h, m+h


'''
def compare_reco_confidence_interval(fom_list, reco_op_1, reco_op_2, data_list,
                                     phantom, conf_level=0.95):
    fom_vals_1 = fom_eval(fom_list, reco_op_1, data_list, phantom)
    fom_vals_2 = fom_eval(fom_list, reco_op_2, data_list, phantom)

    #print(fom_vals_1)

    return confidence_interval_t_dist(fom_vals_1-fom_vals_2, conf_level,
                                      axis=1)
'''


def compare_reco_matrix(fom_list, reco_op_list, data_list, phantom,
                        conf_level=0.95):
    """Performs statistical test based on Student's t-test.

    Parameters
    ----------
    fom_list : list
        List of callable where each element corresponds to one FOM.
    reco_op_list : list
        List of reconstruction operator, that each takes data as input and
        outputs a reconstruction.
    data_list : list
        List of data sets.
    phantom : `FnBase` or `ProductSpace`
        The phantom with respect to which the FOM's are computed.
    conf_level : float, optional
        Value that defines the confidnece level.

    Returns
    -------
    confidence_vals : numpy arrays
        Contains matrices. Each matrix corresponds a FOM in fom_list. For i<j,
        the ij:th entry is positive if the entire confidence interval is
        positive, negative if the confidence intenval is strictly negative,
        and zero if the confidence interval contains zero. Note that the
        confidence interval is defined for the difference between the FOM-value
        of the i:th and j:th reconstruction operator, respectively.
    """
    num_reco = len(reco_op_list)
    num_fom = len(fom_list)
    num_data = len(data_list)

    confidence_vals = np.zeros([num_fom, num_reco, num_reco])
    fom_vals = np.zeros([num_reco, num_fom, num_data])

    for reco_op, i in zip(reco_op_list, range(num_reco)):
        fom_vals[i, :, :] = fom_eval(fom_list, reco_op, data_list, phantom)

    for i in range(num_reco):
        for j in range(num_reco):
            if i < j:
                # Compare reconstruction method i with j using Student's
                # t-distribution
                x_mean, x_min, x_max = confidence_interval_t_dist(
                    fom_vals[i, :, :]-fom_vals[j, :, :], conf_level, axis=1)

                # Display the distance between 0 and the confidence interval
                # (with minus if confidence interval is strictly negative)
                # A positive value means that the reco method i is better than
                # reco method j.
                confidence_vals[:, i, j] = np.median([np.zeros(num_fom),
                                                      x_min, x_max], axis=0)
                confidence_vals[:, j, i] = -np.sign(np.median(
                    [np.zeros(num_fom), x_min, x_max], axis=0))

    return confidence_vals
