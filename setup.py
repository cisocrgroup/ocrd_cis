"""
Installs:
    - ocrd-cis-align
    - ocrd-cis-profile
    - ocrd-cis-ocropy-clip
    - ocrd-cis-ocropy-denoise
    - ocrd-cis-ocropy-deskew
    - ocrd-cis-ocropy-binarize
    - ocrd-cis-ocropy-resegment
    - ocrd-cis-ocropy-segment
    - ocrd-cis-ocropy-dewarp
    - ocrd-cis-ocropy-recognize
    - ocrd-cis-ocropy-train
    - ocrd-cis-aio
    - ocrd-cis-stats
    - ocrd-cis-lang
    - ocrd-cis-clean
    - ocrd-cis-cutter
    - ocrd-cis-importer
"""

import codecs
from setuptools import setup
from setuptools import find_packages

with codecs.open('README.md', encoding='utf-8') as f:
    README = f.read()

setup(
    include_package_data = True,
    name='cis-ocrd',
    version='0.0.4',
    description='CIS OCR-D command line tools',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Florian Fink, Tobias Englmeier, Christoph Weber',
    author_email='finkf@cis.lmu.de, englmeier@cis.lmu.de, web_chris@msn.com',
    url='https://github.com/cisocrgroup/cis-ocrd-py',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'ocrd>=2.0.0a1',
        'click',
        'scipy',
        'numpy>=1.17.0',
        'pillow>=6.2.0',
        'matplotlib>3.0.0',
        'python-Levenshtein',
        'calamari_ocr'
    ],
    package_data={
        '': ['*.json', '*.yml', '*.yaml'],
        'ocrd_cis': ['ocrd_cis/jar/ocrd-cis.jar'],
    },
    entry_points={
        'console_scripts': [
            'ocrd-cis-align=ocrd_cis.align.cli:cis_ocrd_align',
            'ocrd-cis-profile=ocrd_cis.profile.cli:cis_ocrd_profile',
            'ocrd-cis-ocropy-binarize=ocrd_cis.ocropy.cli:cis_ocrd_ocropy_binarize',
            'ocrd-cis-ocropy-clip=ocrd_cis.ocropy.cli:cis_ocrd_ocropy_clip',
            'ocrd-cis-ocropy-denoise=ocrd_cis.ocropy.cli:cis_ocrd_ocropy_denoise',
            'ocrd-cis-ocropy-deskew=ocrd_cis.ocropy.cli:cis_ocrd_ocropy_deskew',
            'ocrd-cis-ocropy-dewarp=ocrd_cis.ocropy.cli:cis_ocrd_ocropy_dewarp',
            'ocrd-cis-ocropy-recognize=ocrd_cis.ocropy.cli:cis_ocrd_ocropy_recognize',
            'ocrd-cis-ocropy-rec=ocrd_cis.ocropy.cli:cis_ocrd_ocropy_rec',
            'ocrd-cis-ocropy-resegment=ocrd_cis.ocropy.cli:cis_ocrd_ocropy_resegment',
            'ocrd-cis-ocropy-segment=ocrd_cis.ocropy.cli:cis_ocrd_ocropy_segment',
            'ocrd-cis-ocropy-train=ocrd_cis.ocropy.cli:cis_ocrd_ocropy_train',
            'ocrd-cis-aio=ocrd_cis.aio.cli:cis_ocrd_aio',
            'ocrd-cis-stats=ocrd_cis.div.cli:cis_ocrd_stats',
            'ocrd-cis-lang=ocrd_cis.div.cli:cis_ocrd_lang',
            'ocrd-cis-clean=ocrd_cis.div.cli:cis_ocrd_clean',
            'ocrd-cis-importer=ocrd_cis.div.cli:cis_ocrd_importer',
            'ocrd-cis-cutter=ocrd_cis.div.cli:cis_ocrd_cutter',
        ]
    },
)
