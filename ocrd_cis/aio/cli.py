import click, sys

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_cis.aio.aio import AllInOne


@click.command()
@ocrd_cli_options
def cis_ocrd_aio(*args, **kwargs):
    return AllInOne()