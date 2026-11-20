import gc
import itertools
import os
import threading
import time
import traceback

import torch


_GB = 1024 ** 3


def _rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / _GB
    except ImportError:
        with open("/proc/self/statm") as file:
            return int(file.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / _GB


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _table(title, headers, rows):
    print(f"\n### {title}")
    print(f"| {' | '.join(headers)} |")
    print(f"|{'|'.join(['---'] + ['---:'] * (len(headers) - 1))}|")
    for row in rows:
        print(f"| {' | '.join(row)} |")


class ModelService:
    def run_tests(self, device=None, repeats=3, deploy_modes=None):
        """Benchmark deploy modes until `NotImplementedError` and print two tables.

        Args:
            deploy_modes: optional list of modes to run. By default every mode is
                tried until one raises `NotImplementedError`.
        """
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        selected_modes = list(deploy_modes) if deploy_modes is not None else None
        results = {}
        try:
            for mode in (selected_modes if selected_modes is not None else itertools.count()):
                print(f"\ndeploy_mode {mode} ...", flush=True)
                try:
                    load = self._measure(device, self.load_models, device=str(device), deploy_mode=mode)
                except NotImplementedError:
                    print(f"deploy_mode {mode} is not implemented.")
                    if selected_modes is None:
                        break
                    continue
                except Exception as error:
                    # Keep benchmarking the remaining deploy modes: a mode that
                    # does not fit in VRAM must not hide a working fallback.
                    traceback.print_exc()
                    print(f"deploy_mode {mode} failed to load: {error!r}", flush=True)
                    results[mode] = {"idle": None, "load": None, "runs": {}, "error": repr(error)}
                    self.pipe = None
                    continue
                idle = self._snapshot(device)
                runs = {}
                try:
                    for repeat in range(1, repeats + 1):
                        inputs = self.example_inputs()
                        if isinstance(inputs, dict):
                            inputs = [inputs]
                        for sample, kwargs in enumerate(inputs):
                            output = {}

                            def run_inference(kwargs=kwargs, output=output):
                                output["value"] = self.infer(**kwargs)

                            runs[sample, repeat] = self._measure(device, run_inference)
                            self._save_output(output.get("value"), mode, sample, repeat)
                except Exception as error:
                    traceback.print_exc()
                    print(f"deploy_mode {mode} failed during inference: {error!r}", flush=True)
                    results[mode] = {"idle": idle, "load": load, "runs": runs, "error": repr(error)}
                    self.pipe = None
                    continue
                results[mode] = {"idle": idle, "load": load, "runs": runs}
                self.pipe = None  # release the models before the next deploy mode
        finally:
            self.pipe = None
            self._report(results)
        if results and all("error" in result for result in results.values()):
            raise RuntimeError("every deploy mode failed; see the log above")
        return results

    def _measure(self, target, function, **kwargs):
        # Returns (peak allocated VRAM GB, peak RSS GB, elapsed s).
        gc.collect()
        _sync(target)
        if target.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(target)
        peak, stop = 0.0, threading.Event()

        def sample():
            nonlocal peak
            while not stop.wait(0.01):
                peak = max(peak, _rss_gb())

        threading.Thread(target=sample, daemon=True).start()
        start = time.perf_counter()
        try:
            function(**kwargs)
            _sync(target)
            elapsed = time.perf_counter() - start
        finally:
            stop.set()
            peak = max(peak, _rss_gb())
        if target.type == "cuda":
            return (torch.cuda.max_memory_allocated(target) / _GB, peak, elapsed)
        return None, peak, elapsed

    def _snapshot(self, target):
        # Returns (allocated VRAM GB, RSS GB) at the current instant.
        _sync(target)
        rss = _rss_gb()
        if target.type == "cuda":
            return torch.cuda.memory_allocated(target) / _GB, rss
        return None, rss

    def _save_output(self, output, mode, sample, repeat):
        if output is None:
            return
        directory = os.path.join(".", "data", "diffsynth_service_tests")
        os.makedirs(directory, exist_ok=True)
        stem = f"{self.__class__.__name__}_mode{mode}_sample{sample}_run{repeat}"
        self._save_value(output, os.path.join(directory, stem))

    def _save_value(self, value, path):
        from PIL import Image
        if isinstance(value, Image.Image):
            value.save(f"{path}.png")
        elif isinstance(value, torch.Tensor):
            torch.save(value, f"{path}.pt")
        elif isinstance(value, dict):
            for key, item in value.items():
                self._save_value(item, f"{path}_{str(key).replace(os.sep, '_')}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self._save_value(item, f"{path}_{index}")
        elif hasattr(value, "save"):
            value.save(f"{path}.png")
        else:
            torch.save(value, f"{path}.pt")

    def _report(self, results):
        if not results:
            print("No deploy mode is implemented.")
            return
        modes = list(results)
        keys = sorted({key for result in results.values() for key in result["runs"]})

        def fmt(value):
            return "-" if value is None else f"{value:.1f}"

        def label(key):
            return f"Sample {key[0]} \u00b7 run {key[1]}"

        headers = ["Phase"]
        headers += [f"mode {mode} VRAM GB" for mode in modes]
        headers += [f"mode {mode} Memory GB" for mode in modes]

        def memory_row(name, getter):
            row = [name]
            for mode in modes:
                value = getter(mode)
                row += ["-" if value is None else fmt(value[0])]
            for mode in modes:
                value = getter(mode)
                row += ["-" if value is None else fmt(value[1])]
            return row

        memory_rows = [
            memory_row(label(key), lambda mode, key=key: results[mode]["runs"].get(key)) for key in keys
        ]

        inference_peak = {}
        maximum = {}
        for mode in modes:
            records = list(results[mode]["runs"].values())
            inference_peak[mode] = tuple(
                max((record[index] for record in records if record[index] is not None), default=None)
                for index in range(2)
            )
            all_records = [results[mode]["idle"], results[mode]["load"]] + records
            maximum[mode] = tuple(
                max((record[index] for record in all_records if record is not None and record[index] is not None), default=None)
                for index in range(2)
            )
        memory_rows += [
            memory_row("Waiting", lambda mode: results[mode]["idle"]),
            memory_row("Loading", lambda mode: results[mode]["load"]),
            memory_row("Inference", lambda mode: inference_peak[mode]),
            memory_row("Maximum", lambda mode: maximum[mode]),
        ]
        _table("Table 1. Peak memory and VRAM per phase", headers, memory_rows)

        def average(mode, sample=None):
            times = [record[2] for (current, repeat), record in results[mode]["runs"].items()
                     if repeat > 1 and (sample is None or current == sample)]
            return fmt(sum(times) / len(times)) if times else "-"

        samples = sorted({key[0] for key in keys})
        time_headers = ["Sample \u00b7 run"] + [f"mode {mode} Time s" for mode in modes]

        time_rows = [
            [label(key)] + [
                fmt(results[mode]["runs"][key][2]) if key in results[mode]["runs"] else "-"
                for mode in modes
            ] for key in keys
        ]
        for sample in samples:
            time_rows.append([f"Sample {sample} average"] + [average(mode, sample) for mode in modes])
        time_rows.append(["Average"] + [average(mode) for mode in modes])
        _table("Table 2. Inference latency in seconds", time_headers, time_rows)
        for mode in modes:
            error = results[mode].get("error")
            if error is not None:
                print(f"\n> deploy_mode {mode} failed: {error}")

    def example_inputs(self):
        raise NotImplementedError("`example_inputs` is not implemented.")

    def load_models(self, device="cuda", deploy_mode=0):
        raise NotImplementedError("`load_models` is not implemented.")

    def infer(self, **kwargs):
        raise NotImplementedError("`infer` is not implemented.")
