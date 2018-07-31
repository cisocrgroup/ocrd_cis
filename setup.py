"""
Installs:
    - ocrd-cis-align
"""

from setuptools import setup, find_packages

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
        'ocrd == 0.7.0',
        'click',
    ],
    package_data={
        '': ['*.json', '*.yml', '*.yaml'],
    },
    entry_points={
        'console_scripts': [
            'ocrd-cis-align=align.cli:cis_ocrd_align'
        ]
    },
)
