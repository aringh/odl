# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test for the figures of merit (FOMs) that use a known ground truth."""

from __future__ import division
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


if __name__ == '__main__':
    odl.util.test_file(__file__)
