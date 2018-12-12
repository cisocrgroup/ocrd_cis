"""
Installs:
    - ocrd-cis-align
    - ocrd-cis-profile
    - ocrd-cis-train
    - ocrd-cis-ocropy-recognize
    - ocrd-cis-ocropy-train
    - ocrd-cis-aio
    - ocrd-cis-stats
    - ocrd-cis-lang
    - ocrd-cis-clean
"""

from setuptools import setup
from setuptools import find_packages

setup(
    name='cis-ocrd',
    version='0.0.1',
    description='description',
    long_description='long description',
    author='Florian Fink, Tobias Englmeier, Christoph Weber',
    author_email='finkf@cis.lmu.de, englmeier@cis.lmu.de, web_chris@msn.com',
    url='https://github.com/cisocrgroup/cis-ocrd-py',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'ocrd>=0.7.2',
        'click',
        'scipy',
        'matplotlib',
        'python-Levenshtein'
    ],
    package_data={
        '': ['*.json', '*.yml', '*.yaml'],
    },
    entry_points={
        'console_scripts': [
            'ocrd-cis-align=ocrd_cis.align.cli:cis_ocrd_align',
            'ocrd-cis-profile=ocrd_cis.profile.cli:cis_ocrd_profile',
            'ocrd-cis-train=ocrd_cis.train.trainer:cis_ocrd_train',
            'ocrd-cis-ocropy-recognize=ocrd_cis.ocropy.cli:cis_ocrd_ocropy_recognize',
            'ocrd-cis-ocropy-train=ocrd_cis.ocropy.cli:cis_ocrd_ocropy_train',
            'ocrd-cis-aio=ocrd_cis.aio.cli:cis_ocrd_aio',
            'ocrd-cis-stats=ocrd_cis.div.cli:cis_ocrd_stats',
            'ocrd-cis-lang=ocrd_cis.div.cli:cis_ocrd_lang',
            'ocrd-cis-clean=ocrd_cis.div.cli:cis_ocrd_clean',
        ]
    },
)
