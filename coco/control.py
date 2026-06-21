"""The Constitutional Controller for navigating constrained and uncertain environments."""

#
# Copyright (c) Simon Kohaut, Felix Divo, and contributors
#
# This file is part of CoCo and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with CoCo.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Standard Library
from collections.abc import Callable
from copy import deepcopy

# Third Party
import torch
from numpy import array, mean
from numpy.typing import NDArray
from promis.geo import Collection


class ConstitutionalController:
    def apply_doubt(
        self,
        landscape: Collection,
        doubt_density: Callable[[int], NDArray],
        doubt_space: dict[str, dict],
        number_of_samples: int,
        scaler=1000.0,
    ) -> Collection:
        interpolator = landscape.get_interpolator("hybrid")
        samples = scaler * doubt_density.sample(number_of_samples, doubt_space)[0]

        doubtful_landscape = deepcopy(landscape)
        doubtful_landscape.data['v0'] = [
            mean(interpolator(location_samples.detach().numpy()))
            for location_samples in torch.from_numpy(doubtful_landscape.coordinates()[:, None, :]) + samples[None, :, :]
        ]

        return doubtful_landscape

    def compliance(
        self,
        path: NDArray,
        landscape: Collection,
        doubt_density: Callable[[int], NDArray],
        doubt_space: dict[str, dict],
        number_of_samples: int,
    ) -> NDArray:
        interpolator = landscape.get_interpolator("hybrid")
        samples = doubt_density.sample(number_of_samples, doubt_space)[0]

        compliances = array(
            [
                mean(interpolator(location_samples.detach().numpy()))
                for location_samples in torch.from_numpy(path[:, None, :]) + samples[None, :, :]
            ]
        )

        return compliances