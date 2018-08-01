"""
Installs:
    - ocrd-cis-align
    - ocrd-cis-profile
"""

from setuptools import setup
from setuptools import find_packages

setup(
    name='cis-ocrd',
    version='0.0.1',
    description='description',
    long_description='long description',
    author='Florian Fink, Tobias Englmeier',
    author_email='finkf@cis.lmu.de, englmeier@cis.lmu.de',
    url='https://github.com/cisocrgroup/cis-ocrd-py',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'ocrd==0.7.0',
        'click',
    ],
    package_data={
        '': ['*.json', '*.yml', '*.yaml'],
    },
    entry_points={
        'console_scripts': [
            'ocrd-cis-align=ocrd_cis.align.cli:cis_ocrd_align',
            'ocrd-cis-profile=ocrd_cis.profile.cli:cis_ocrd_profile',
        ]
    },
)
