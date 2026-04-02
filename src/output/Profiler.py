import time
import torch
from contextlib import contextmanager
from output.Log import Log

log = Log.for_source(__name__)


class Profiler:
    def __init__(self, name: str, report_freq: int = 100, device: torch.device = None):
        self.name = name
        self.report_freq = report_freq
        self.device = device
        self.metrics = {
            "data_loading": [],
            "transfer": [],
            "forward": [],
            "backward": [],
            "optimizer": [],
        }
        self.iter_start_time = time.perf_counter()

    def _synchronize(self):
        if self.device and self.device.type == "cuda":
            torch.cuda.synchronize()

    @contextmanager
    def measure(self, name: str):
        start = time.perf_counter()
        yield
        self._synchronize()
        duration = time.perf_counter() - start
        if name in self.metrics:
            self.metrics[name].append(duration)

    def start_iteration(self):
        duration = time.perf_counter() - self.iter_start_time
        self.metrics["data_loading"].append(duration)

    def end_iteration(self):
        self.iter_start_time = time.perf_counter()

    def report(self, batch_idx: int, total_batches: int, **kwargs):
        if (batch_idx + 1) % self.report_freq == 0 or (batch_idx + 1) == total_batches:
            stats = {}
            total_gpu_ms = 0

            for name, times in self.metrics.items():
                if times:
                    count = min(len(times), self.report_freq)
                    avg_ms = (sum(times[-count:]) / count) * 1000
                    stats[f"avg_{name}_ms"] = f"{avg_ms:.2f}"

                    if name != "data_loading":
                        total_gpu_ms += avg_ms

            stats["avg_batch_gpu_total_ms"] = f"{total_gpu_ms:.2f}"

            log.information(
                f"{self.name}_batch_completed",
                batch=batch_idx + 1,
                total_batches=total_batches,
                **kwargs,
                **stats,
            )
