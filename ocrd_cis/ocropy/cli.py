import click, os

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_cis.ocropy.binarize import OcropyBinarize
from ocrd_cis.ocropy.denoise import OcropyDenoise
from ocrd_cis.ocropy.deskew import OcropyDeskew
from ocrd_cis.ocropy.clip import OcropyClip
from ocrd_cis.ocropy.resegment import OcropyResegment
from ocrd_cis.ocropy.dewarp import OcropyDewarp
from ocrd_cis.ocropy.recognize import OcropyRecognize
from ocrd_cis.ocropy.segment import OcropySegment
from ocrd_cis.ocropy.train import OcropyTrain
from ocrd_cis.ocropy.rec import OcropyRec


@click.command()
@ocrd_cli_options
def ocrd_cis_ocropy_binarize(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcropyBinarize, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_cis_ocropy_deskew(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcropyDeskew, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_cis_ocropy_denoise(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcropyDenoise, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_cis_ocropy_clip(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcropyClip, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_cis_ocropy_resegment(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcropyResegment, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_cis_ocropy_dewarp(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcropyDewarp, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_cis_ocropy_recognize(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcropyRecognize, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_cis_ocropy_segment(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcropySegment, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_cis_ocropy_train(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcropyTrain, *args, **kwargs)

@click.command()
@click.option('-f', '--fromdir', default = os.getcwd())
@click.option('-t', '--todir', default = os.getcwd())
@click.option('-m', '--model', default='en-default.pyrnn.gz')
def ocrd_cis_ocropy_rec(fromdir, todir , model):
    return OcropyRec(fromdir, todir, model)
