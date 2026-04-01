import bionix_ml.uteis.arquivo as arquivo_io
import os

# Lightweight one-shot helpers kept for backwards compatibility.
import diagnostics.logger as logger

fn debug_marker(var desc: String, var context: String):
    # Deprecated wrapper: forward to centralized logger.marker
    try:
        logger.marker(desc, context)
    except _:
        pass

fn debug_marker_verbose(var desc: String, var context: String):
    # Deprecated wrapper: forward to centralized logger.marker_verbose
    try:
        logger.marker_verbose(desc, context)
    except _:
        pass

fn debug_error(var desc: String, var context: String):
    try:
        logger.error_log(desc, context)
    except _:
        pass
