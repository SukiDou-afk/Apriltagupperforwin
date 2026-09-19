from __future__ import annotations
import logging
import threading
import time

class EventHook:
    """Callbacks run on the emitting worker thread. Keep handlers short."""
    def __init__(self):
        self._handlers = []
        self._lock = threading.Lock()
    def connect(self, callback):
        with self._lock:
            self._handlers.append(callback)
    def disconnect(self, callback):
        with self._lock:
            self._handlers.remove(callback)
    def emit(self, *args):
        with self._lock:
            handlers = tuple(self._handlers)
        for handler in handlers:
            try:
                handler(*args)
            except Exception:
                logging.exception("SDK event callback failed")

class ThreadWorker(threading.Thread):
    def __init__(self, parent=None):
        super().__init__(daemon=True)
    def isRunning(self):
        return self.is_alive()
    def wait(self, milliseconds=None):
        if self.ident is None or threading.current_thread() is self:
            return not self.is_alive()
        self.join(None if milliseconds is None else milliseconds / 1000.)
        return not self.is_alive()
    @staticmethod
    def msleep(milliseconds):
        time.sleep(milliseconds / 1000.)
