"""Scalar logger that writes a durable CSV and, when available, TensorBoard.

On Kaggle the session ends and the container is discarded, so the CSV is the
artifact that actually survives into the output dataset.  TensorBoard is
optional: if ``torch.utils.tensorboard`` is missing, logging degrades to
CSV-only rather than failing the run.
"""
from __future__ import annotations

import csv
import pathlib
import time

try:
    from torch.utils.tensorboard import SummaryWriter as _TBWriter
except Exception:  # pragma: no cover - tensorboard is optional
    _TBWriter = None


class CsvSummaryWriter:
    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        self.log_dir = str(log_dir)
        path = pathlib.Path(self.log_dir)
        path.mkdir(parents=True, exist_ok=True)

        self._tb = None
        if use_tensorboard and _TBWriter is not None:
            try:
                self._tb = _TBWriter(self.log_dir)
            except Exception:
                self._tb = None

        csv_path = path / "scalars.csv"
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        self._file = csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if write_header:
            self._writer.writerow(["wall_time", "step", "tag", "value"])
            self._file.flush()
        self._rows_since_flush = 0
        self._closed = False
        self._warned = False

    @staticmethod
    def _to_float(value) -> float:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        try:
            value = self._to_float(scalar_value)
        except Exception:
            return
        if self._tb is not None:
            try:
                self._tb.add_scalar(tag, value, global_step=global_step, walltime=walltime)
            except Exception:
                self._tb = None
        try:
            self._writer.writerow(
                [
                    time.time() if walltime is None else float(walltime),
                    "" if global_step is None else int(global_step),
                    str(tag),
                    value,
                ]
            )
            self._rows_since_flush += 1
            if self._rows_since_flush >= 64:
                self.flush()
        except Exception as exc:  # logging must never break training
            if not self._warned:
                print(f"[csv logging] scalar mirror disabled for a row: {exc}")
                self._warned = True

    def add_text(self, tag, text, global_step=None):
        if self._tb is not None:
            try:
                self._tb.add_text(tag, str(text), global_step=global_step)
            except Exception:
                self._tb = None
        try:
            path = pathlib.Path(self.log_dir) / "text_log.jsonl"
            with path.open("a", encoding="utf-8") as f:
                import json

                f.write(
                    json.dumps(
                        {"step": global_step, "tag": str(tag), "text": str(text)}
                    )
                    + "\n"
                )
        except Exception:
            pass

    def flush(self):
        if not self._closed:
            try:
                self._file.flush()
                self._rows_since_flush = 0
            except Exception:
                pass
        if self._tb is not None:
            try:
                self._tb.flush()
            except Exception:
                pass

    def close(self):
        if self._closed:
            return
        self.flush()
        try:
            self._file.close()
        finally:
            self._closed = True
        if self._tb is not None:
            try:
                self._tb.close()
            except Exception:
                pass
