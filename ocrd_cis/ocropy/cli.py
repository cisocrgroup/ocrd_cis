import click, os

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_cis.ocropy.recognize import OcropyRecognize
from ocrd_cis.ocropy.train import OcropyTrain
from ocrd_cis.ocropy.rec import OcropyRec



@click.command()
@ocrd_cli_options
def cis_ocrd_ocropy_recognize(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcropyRecognize, *args, **kwargs)

@click.command()
@ocrd_cli_options
def cis_ocrd_ocropy_train(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcropyTrain, *args, **kwargs)

@click.command()
@click.option('-f', '--fromdir', default = os.getcwd())
@click.option('-t', '--todir', default = os.getcwd())
@click.option('-m', '--model', default='en-default.pyrnn.gz')
def cis_ocrd_ocropy_rec(fromdir, todir , model):
    return OcropyRec(fromdir, todir, model)
