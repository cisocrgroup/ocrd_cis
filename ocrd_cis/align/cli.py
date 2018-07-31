import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_cis.align.aligner import Aligner


@click.command()
@ocrd_cli_options
def cis_ocrd_align(*args, **kwargs):
    # kwargs['cache_enabled'] = False
    return ocrd_cli_wrap_processor(Aligner, *args, **kwargs)
