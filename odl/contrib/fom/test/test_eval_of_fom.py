# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test for the figures of merit (FOMs) that use a known ground truth."""

from __future__ import division
import numpy as np
import pytest
import odl
from odl.contrib import fom
from odl.util.testutils import noise_element, noise_elements


def test_fom_eval():
    space = odl.uniform_discr(0, 1, 11)

    phantom = space.one()
    data_list = [phantom + 0.1 * noise
                 for noise in noise_elements(space, n=2)[1]]
    reco_op = odl.IdentityOperator(space)

    fom_list = [fom.mean_squared_error,
                fom.mean_absolute_error,
                fom.mean_value_difference]

    result = fom.fom_eval(fom_list, reco_op, data_list, phantom)

    for i, current_fom in enumerate(fom_list):
        for j, current_data in enumerate(data_list):
            expected_result = current_fom(reco_op(current_data), phantom)
            assert expected_result == result[i, j]


def test_confidence_interval_t_dist():
    data_1 = np.random.normal(loc=0.0, scale=0.1, size=10)
    data_2 = np.random.normal(loc=1.0, scale=0.1, size=10)

    data = np.array([data_1, data_2])

    means, lower_bounds, upper_bounds = fom.confidence_interval_t_dist(
        data=data, conf_level=0.9, axis=1)

    assert (lower_bounds <= means).all()
    assert (means <= upper_bounds).all()
    assert means == pytest.approx([0, 1], abs=1e-1)


def test_compare_reco_matrix():
    space = odl.uniform_discr(0, 1, 11)

    phantom = space.one()

    num_data = 5  # 5 data sets
    data_list = [phantom + 0.1 * noise
                 for noise in noise_elements(space, n=5)[1]]

    num_reco_op = 2  # test with 2 operators
    reco_opt_1 = odl.IdentityOperator(space)
    reco_opt_2 = odl.IdentityOperator(space) + 0.3 * noise_element(space)
    reco_op_list = [reco_opt_1, reco_opt_2]

    num_foms = 3  # use 3 FOMs
    fom_list = [fom.mean_squared_error,
                fom.mean_absolute_error,
                fom.mean_value_difference]

    result = fom.compare_reco_matrix(fom_list, reco_op_list, data_list,
                                     phantom, conf_level=0.9)

    # Correct dimension on output
    assert result.shape == (num_foms, num_reco_op, num_reco_op)

    # Compute all fom for each reco operator on each data
    fom_values = np.empty((num_reco_op, num_foms, num_data))
    for i, reco_op in enumerate(reco_op_list):
        for j, current_fom in enumerate(fom_list):
            for k, data in enumerate(data_list):
                fom_values[i, j, k] = current_fom(reco_op(data), phantom)

    # Take the difference between the foms from reco operator 1 and 2
    fom_diffs = np.empty((num_foms, num_data))
    for j in range(num_foms):
        fom_diffs[j, :] = fom_values[0, j, :] - fom_values[1, j, :]

    # Compute confidence intervals around these differences
    means, lower_bounds, upper_bounds = fom.confidence_interval_t_dist(
        data=fom_diffs, conf_level=0.9, axis=1)

    # Check that the conslusion is the same as computed with function
    conclusion_value = np.empty(3)
    for i in range(num_foms):
        conclusion_value[i] = result[i, 0, 1]

    # FOMs where "rec_op 1 better than 2" agrees
    assert ((upper_bounds < 0) == (conclusion_value < 0)).all()

    # FOMs where "rec_op 1 worse than 2" agrees
    assert ((lower_bounds > 0) == (conclusion_value > 0)).all()

    # FOMs which are inconclusive agrees
    inconclusive = [val_1*val_2 for
                    val_1, val_2 in zip(lower_bounds < 0, upper_bounds > 0)]
    assert (inconclusive == (conclusion_value == 0)).all()


if __name__ == '__main__':
    odl.util.test_file(__file__)
