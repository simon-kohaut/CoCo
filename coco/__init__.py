"""CoCo - Constitutional Control for constrained navigation tasks."""

#
# Copyright (c) Simon Kohaut, Felix Divo, and contributors
#
# This file is part of CoCo and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with CoCo.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# CoCo
from control import ConstitutionalController
from doubt import DoubtDensity

__all__ = ["ConstitutionalController", "DoubtDensity"]
__version__ = "0.0.1"
__author__ = "Simon Kohaut"


def get_author():
    return __author__


def get_version():
    return __version__