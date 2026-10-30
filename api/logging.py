"""Structured web diagnostics without request bodies, passwords or patient values."""
import json
import logging
from datetime import datetime, timezone


class WebFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({"time":datetime.now(timezone.utc).isoformat(), "level":record.levelname,
                           "event":record.getMessage(), "request_id":getattr(record,"request_id",None),
                           "route":getattr(record,"route",None), "status":getattr(record,"status",None),
                           "duration_ms":getattr(record,"duration_ms",None),
                           "exception_type":record.exc_info[0].__name__ if record.exc_info else None,
                           "exception_message":str(record.exc_info[1]) if record.exc_info else None})


def configure_web_logging():
    logger = logging.getLogger("ta_efl.web")
    if not any(isinstance(h.formatter, WebFormatter) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(WebFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
