"""Minimal structured (JSONL) logging for the defense stack.

Decision-level events only — never call at physics rate.
Usage:
    from gym_pybullet_drones.utils.tlog import get_logger
    log = get_logger("strategy")
    log.log("assignment", defender=3, enemy_uid=7, cost=1.23)
"""
import json
import os
import threading
import time


class JsonlLogger:
    _lock = threading.Lock()
    _file = None
    _path = None
    _console = True

    @classmethod
    def init(cls, run_dir=None, console=True):
        """Call once at startup. Defaults to runs/<ts>/sim.jsonl"""
        with cls._lock:
            cls._console = console
            if cls._file is not None:
                return cls._path
            if run_dir is None:
                run_dir = os.path.join("runs", time.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(run_dir, exist_ok=True)
            cls._path = os.path.join(run_dir, "sim.jsonl")
            cls._file = open(cls._path, "a", encoding="utf-8")
            return cls._path

    @classmethod
    def write(cls, record):
        with cls._lock:
            if cls._file is None:
                cls.init()
            line = json.dumps(record, default=float) + "\n"
            try:
                cls._file.write(line)
                cls._file.flush()
            except OSError:
                # Windows can transiently invalidate the handle (AV scan,
                # errno 22). Reopen once and continue — never crash the sim.
                try:
                    cls._file.close()
                except Exception:
                    pass
                cls._file = None
                cls.init(run_dir=os.path.dirname(cls._path or "") or None)
                try:
                    cls._file.write(line)
                    cls._file.flush()
                except OSError:
                    pass  # drop the record rather than take down the run
            if cls._console:
                print(f"[{record['event']}]", {k: v for k, v in record.items()
                                                if k not in ("t", "event")})

    @classmethod
    def close(cls):
        with cls._lock:
            if cls._file is not None:
                cls._file.close()
                cls._file = None


class _Logger:
    def __init__(self, source):
        self.source = source

    def log(self, event, **fields):
        record = {"t": round(time.time(), 6), "source": self.source,
                  "event": event}
        record.update(fields)
        JsonlLogger.write(record)

    # Convenience aliases
    def info(self, event, **fields): self.log(event, level="info", **fields)
    def warn(self, event, **fields): self.log(event, level="warn", **fields)


def get_logger(source):
    return _Logger(source)
