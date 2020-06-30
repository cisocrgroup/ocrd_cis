import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_cis.div.stats import Stats
from ocrd_cis.div.lang import Lang
from ocrd_cis.div.cutter import Cutter
from ocrd_cis.div.importer import Importer


@click.command()
@ocrd_cli_options
def ocrd_cis_stats(*args, **kwargs):
    return ocrd_cli_wrap_processor(Stats, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_cis_lang(*args, **kwargs):
    return ocrd_cli_wrap_processor(Lang, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_cis_cutter(*args, **kwargs):
    return ocrd_cli_wrap_processor(Cutter, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_cis_importer(*args, **kwargs):
    return ocrd_cli_wrap_processor(Importer, *args, **kwargs)
