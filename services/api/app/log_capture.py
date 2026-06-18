"""In-memory circular log buffer — exposed via GET /api/logs."""
import collections
import logging
import threading

_buffer: collections.deque = collections.deque(maxlen=300)
_lock = threading.Lock()


class _BufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        with _lock:
            _buffer.append(self.format(record))


def setup() -> None:
    handler = _BufferHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
    )
    logging.root.addHandler(handler)


def get_lines(n: int = 100) -> list[str]:
    with _lock:
        lines = list(_buffer)
    return lines[-n:]
