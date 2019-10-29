import click
import os

from ocrd_cis.aio.aio import AllInOne


@click.command()
@click.option('-p', '--parameter', envvar='PARAMETER_DIR', metavar='PARAMETER_DIR', help='Path to Parameter File', show_default=True)
@click.option('-d', '--directory', envvar='WORKSPACE_DIR', default='.', type=click.Path(file_okay=False), metavar='WORKSPACE_DIR', help='Changes the workspace folder location.', show_default=True)
@click.option('-v', '--verbose', is_flag=True, help='Enables verbose mode.')
@click.option('-l', '--download', is_flag=True, help='downloads and updates all projects automatically.')


def cis_ocrd_aio(directory, parameter, verbose, download):
    if directory == '.':
        directory = os.getcwd()
    return AllInOne(directory, parameter, verbose, download)
