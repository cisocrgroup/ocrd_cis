import click

from ocrd.decorators import ocrd_cli_options
from ocrd.decorators import ocrd_cli_wrap_processor
from ocrd_cis.profile.profiler import Profiler


@click.command()
@ocrd_cli_options
def cis_ocrd_profile(*args, **kwargs):
    return ocrd_cli_wrap_processor(Profiler, *args, **kwargs)
