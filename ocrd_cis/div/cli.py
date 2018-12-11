import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_cis.div.stats import Stats
from ocrd_cis.div.lang import Lang


@click.command()
@ocrd_cli_options
def cis_ocrd_stats(*args, **kwargs):
    return ocrd_cli_wrap_processor(Stats, *args, **kwargs)

@click.command()
@ocrd_cli_options
def cis_ocrd_lang(*args, **kwargs):
    return ocrd_cli_wrap_processor(Lang, *args, **kwargs)