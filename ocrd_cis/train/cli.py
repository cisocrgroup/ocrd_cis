import click
from ocrd.decorators import ocrd_cli_options
from ocrd.decorators import ocrd_cli_wrap_processor
from ocrd_cis.train.trainer import Trainer
import ocrd_cis.train.config as config


@click.command()
@ocrd_cli_options
def cis_ocrd_train(*args, **kwargs):
    if kwargs["log_level"]:
        config.LOG_LEVEL = kwargs["log_level"]
    config.MPATH = kwargs["mets"]
    config.PPATH = kwargs["parameter"]
    return ocrd_cli_wrap_processor(Trainer, *args, **kwargs)
