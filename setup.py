# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

# Package meta-data.
NAME = 'MomSPS'
VERSION = '1.0'
AUTHOR = 'Dimitris Oikonomou'
EMAIL = 'doikono1@jh.edu'
DESCRIPTION = 'Stochastic Polyak Step-sizes and Momentum: Convergence Guarantees and Practical Performance'
URL = 'https://github.com/dimitris-oik/MomSPS'
REQUIRES_PYTHON = '>=3.8.0'
REQUIRED = [
    "torch",
    "numpy",
]


setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    packages=setuptools.find_packages(),
    install_requires=REQUIRED,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=REQUIRES_PYTHON,
)
