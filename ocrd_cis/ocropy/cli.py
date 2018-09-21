import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_cis.ocropy.recognize import OcropyRecognize


@click.command()
@ocrd_cli_options
def cis_ocrd_ocropy_recognize(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcropyRecognize, *args, **kwargs)