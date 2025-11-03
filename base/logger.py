# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import logging
import os

def setup_logging():
    """Configure logging for the application."""
    LOG_LEVEL = os.getenv("LOGLEVEL", "INFO").upper()
    FMT = "%(asctime)s | %(levelname)s | %(message)s"

    # Configure root logger if it doesn't have handlers
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(format=FMT, level=LOG_LEVEL)
    else:
        root_logger.setLevel(LOG_LEVEL)
        for h in root_logger.handlers:
            h.setFormatter(logging.Formatter(FMT))

    # Create and return the application logger
    logger = logging.getLogger("agentic_sdk") 
    return logger
