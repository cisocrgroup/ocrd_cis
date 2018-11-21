import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_cis.stats.stats import Stats



@click.command()
@ocrd_cli_options
def cis_ocrd_stats(*args, **kwargs):
    return ocrd_cli_wrap_processor(Stats, *args, **kwargs)