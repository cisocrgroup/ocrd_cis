import click, sys, os

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_cis.aio.aio import AllInOne


@click.command()
@click.option('-p', '--parameter', envvar='PARAMETER_DIR', metavar='PARAMETER_DIR', help='Path to Parameter File', show_default=True)
@click.option('-d', '--directory', envvar='WORKSPACE_DIR', default='.', type=click.Path(file_okay=False), metavar='WORKSPACE_DIR', help='Changes the workspace folder location.', show_default=True)
def cis_ocrd_aio(directory, parameter):
    if directory == '.': directory = os.getcwd() 
    return AllInOne(directory,  parameter)