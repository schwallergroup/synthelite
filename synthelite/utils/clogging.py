""" Module containing routines to setup proper logging
"""
# pylint: disable=ungrouped-imports, wrong-import-order, wrong-import-position, unused-import
import logging.config
import os

import yaml
import weave
from weave.trace.autopatch import AutopatchSettings, IntegrationSettings

# See Github issue 30 why sklearn is imported here
try:
    import sklearn  # noqa
except ImportError:
    pass
from rdkit import RDLogger

from synthelite.utils.paths import data_path
from synthelite.utils.type_utils import Optional

# Suppress RDKit errors due to incomplete template (e.g. aromatic non-ring atoms)
rd_logger = RDLogger.logger()
rd_logger.setLevel(RDLogger.CRITICAL)


def init_weave(name="synthelite"):
    # Disable integrations that are incompatible with the currently installed SDKs.
    autopatch = AutopatchSettings(cohere=IntegrationSettings(enabled=False))
    weave.init(name, autopatch_settings=autopatch)


def logger() -> logging.Logger:
    """
    Returns the logger that should be used by all classes

    :return: the logger object
    """
    return logging.getLogger("synthelite")


def setup_logger(
    console_level: int, file_level: Optional[int] = None
) -> logging.Logger:
    """
    Setup the logger that should be used by all classes

    The logger configuration is read from the `logging.yml` file.

    :param console_level: the level of logging to the console
    :param file_level: the level of logging to file, if not set logging to file is disabled, default to None
    :return: the logger object
    """
    filename = os.path.join(data_path(), "logging.yml")
    with open(filename, "r") as fileobj:
        config = yaml.load(fileobj.read(), Loader=yaml.SafeLoader)

    config["handlers"]["console"]["level"] = console_level
    if file_level:
        config["handlers"]["file"]["level"] = file_level
    else:
        del config["handlers"]["file"]
        config["loggers"]["synthelite"]["handlers"].remove("file")

    logging.config.dictConfig(config)
    return logger()


def init_logger(log_file: Optional[str] = None):
    logger_obj = logger()
    logger_obj.setLevel("INFO")

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel("INFO")

        formatter = logging.Formatter(
            fmt="%(process)d - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)

        logger_obj.addHandler(file_handler)

    return logger_obj
