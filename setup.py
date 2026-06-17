"""Sets up the CoCO package for installation."""

#
# Copyright (c) Simon Kohaut, Felix Divo, and contributors
#
# This file is part of CoCO and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with CoCO.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

import re

import setuptools

# Find CoCo version and author strings
with open("coco/__init__.py", encoding="utf8") as fd:
    content = fd.read()
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE).group(1)
    author = re.search(r'^__author__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE).group(1)

# Import readme
with open("README.md", encoding="utf8") as readme:
    long_description = readme.read()

setuptools.setup(
    name="coco",
    version=version,
    author=author,
    author_email="simon.kohaut@cs.tu-darmstadt.de",
    description="A Python package for constitutional control in constrained navigation tasks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    package_data={
        "coco": ["py.typed"],  # https://www.python.org/dev/peps/pep-0561/
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        # probabilistic logic in mission design
        "promis",
        # probabilistic modelling of doubt density
        "torch",
        "nflows",
    ],
    extras_require={
        # Building the documentation locally with sphinx
        "doc": [
            "sphinx",
            "nbsphinx",
            "sphinx-markdown-builder",
            "sphinx_rtd_theme",
            "sphinxcontrib-programoutput",
        ],
        # Development tools for quality assurance
        "dev": [
            # static code analysis
            "black",
            "ruff",
            # dynamic code analysis
            "pytest",
        ],
    },
    include_package_data=True,
)