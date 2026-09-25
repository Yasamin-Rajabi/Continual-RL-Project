"""TensorBoard SummaryWriter with a CSV mirror for scalar metrics.

Every ``add_scalar`` call is written to ``scalars.csv`` in the same run
folder as the TensorBoard event files.  The CSV is intentionally long-form so
new scalar tags can be added elsewhere without changing this logger.
"""
from __future__ import annotations

import csv
import pathlib
import time

from torch.utils.tensorboard import SummaryWriter


class CsvSummaryWriter(SummaryWriter):
    """Mirror TensorBoard scalar events to a durable long-form CSV file."""

    def __init__(self, log_dir=None, *args, **kwargs):
        super().__init__(log_dir=log_dir, *args, **kwargs)
        csv_path = pathlib.Path(self.log_dir) / "scalars.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        self._scalar_csv_file = csv_path.open("a", newline="", encoding="utf-8")
        self._scalar_csv_writer = csv.writer(self._scalar_csv_file)
        if write_header:
            self._scalar_csv_writer.writerow(["wall_time", "step", "tag", "value"])
            self._scalar_csv_file.flush()
        self._scalar_csv_rows_since_flush = 0
        self._scalar_csv_warning_emitted = False
        self._scalar_csv_closed = False

    @staticmethod
    def _scalar_to_float(value):
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)

    def add_scalar(
        self,
        tag,
        scalar_value,
        global_step=None,
        walltime=None,
        new_style=False,
        double_precision=False,
    ):
        # Preserve TensorBoard behavior exactly; CSV mirroring is secondary.
        result = super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
            double_precision=double_precision,
        )
        try:
            value = self._scalar_to_float(scalar_value)
            row_wall_time = time.time() if walltime is None else float(walltime)
            step = "" if global_step is None else int(global_step)
            self._scalar_csv_writer.writerow([row_wall_time, step, str(tag), value])
            self._scalar_csv_rows_since_flush += 1
            if self._scalar_csv_rows_since_flush >= 64:
                self._scalar_csv_file.flush()
                self._scalar_csv_rows_since_flush = 0
        except Exception as exc:
            # CSV logging must never alter the training/evaluation path.
            if not self._scalar_csv_warning_emitted:
                print(f"[csv logging] scalar mirror disabled for a row: {exc}")
                self._scalar_csv_warning_emitted = True
        return result

    def flush(self):
        if not self._scalar_csv_closed:
            try:
                self._scalar_csv_file.flush()
                self._scalar_csv_rows_since_flush = 0
            except Exception:
                pass
        return super().flush()

    def close(self):
        if self._scalar_csv_closed:
            return super().close()
        try:
            # SummaryWriter.close() flushes through our overridden flush().
            result = super().close()
        finally:
            try:
                self._scalar_csv_file.flush()
            except Exception:
                pass
            try:
                self._scalar_csv_file.close()
            finally:
                self._scalar_csv_closed = True
        return result
