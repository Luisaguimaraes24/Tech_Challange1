"""
src/utils/logger.py
Logger centralizado do projeto.
Todos os módulos importam daqui — zero prints no projeto.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Retorna um logger configurado com formato padronizado.

    Uso:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("mensagem")
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger