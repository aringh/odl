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
from odl.util.testutils import noise_elements


def test_fom_eval():
    space = odl.uniform_discr(0, 1, 11)

    fom_list = [fom.mean_squared_error,
                fom.mean_absolute_error,
                fom.mean_value_difference]

    reco_op = odl.IdentityOperator(space)
    phantom = space.one()
    data_list = [phantom + 0.1 * noise
                 for noise in noise_elements(space, n=2)[1]]

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


if __name__ == '__main__':
    odl.util.test_file(__file__)
