"""Frozen baselines for osspr intake. Prints one JSON object.

Run from the repository root:

    TZ=Asia/Shanghai taskset -c 0 python3 osspr-intake/harness/bench.py parse_time_point
    TZ=Asia/Shanghai taskset -c 0 python3 osspr-intake/harness/bench.py get_china_candidates
    TZ=Asia/Shanghai taskset -c 0 python3 osspr-intake/harness/bench.py phone_prepare_peak
"""
import datetime
import gc
import json
import logging
import random
import statistics
import sys
import textwrap
import time
import tracemalloc
from pathlib import Path

WARMUP = 2
REPEATS = 9
ROOT = Path(__file__).resolve().parents[2]


def _noise_pct(samples):
    """Full spread of the samples, as a percent of the median."""
    median = statistics.median(samples)
    if median == 0:
        return 0.0
    return 100.0 * (max(samples) - min(samples)) / median


def _time_cases():
    """The 416 strings the native time-parser test actually parses."""
    source = (ROOT / "test" / "test_time_parser.py").read_text(encoding="utf-8")
    start = source.find("        time_string_list = [")
    end = source.find("        print(max([len(item[0])", start)
    chunk = "_ts_1 = 1623604000\n_ts_2 = 1630480532\n" + textwrap.dedent(
        source[start:end]
    )
    namespace = {"datetime": datetime, "time": time}
    exec(chunk, namespace)
    cases = namespace["time_string_list"]
    if len(cases) != 416:
        raise SystemExit(f"expected 416 time cases, found {len(cases)}")
    return [(item[0], item[1]) for item in cases]


def _parse_time_point():
    import jionlp as jio

    logging.getLogger().setLevel(logging.CRITICAL)
    cases = _time_cases()

    def fn():
        parsed = 0
        for text, time_base in cases:
            jio.parse_time(text, time_base=time_base)
            parsed += 1
        return parsed

    return fn, "parsed"


def _china_names(recognizer):
    names = []
    seen = set()
    for admin in recognizer.china_administrative_map_list:
        for name_item in admin:
            for name in name_item:
                if isinstance(name, str) and name not in seen:
                    seen.add(name)
                    names.append(name)
    names.sort()
    return random.Random(0).sample(names, 400)


def _get_china_candidates():
    from jionlp.gadget.location_recognizer import LocationRecognizer

    recognizer = LocationRecognizer()
    recognizer._prepare()
    names = _china_names(recognizer)

    def fn():
        found = 0
        for name in names:
            found += len(recognizer.get_china_candidates(name))
        return found

    return fn, "candidates"


def _phone_prepare_peak():
    from jionlp.gadget.phone_location import PhoneLocation

    def fn():
        gc.collect()
        tracemalloc.start()
        phone = PhoneLocation()
        phone._prepare()
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # Keep the built index live until the peak has been read.
        if phone.cell_phone_location_trie is None:
            raise SystemExit("phone index was not built")
        del phone
        gc.collect()
        return peak

    return fn, "peak_bytes"


def _measure(fn):
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(REPEATS):
        gc.collect()
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    median = statistics.median(samples)
    return {
        "median_ms": median,
        "noise_pct": _noise_pct(samples),
        "samples_ms": samples,
    }


def _measure_peak(fn):
    samples = []
    for _ in range(WARMUP + REPEATS):
        samples.append(fn())
    measured = samples[WARMUP:]
    median = statistics.median(measured)
    return {
        "median_bytes": median,
        "noise_pct": _noise_pct(measured),
        "samples_bytes": measured,
    }


def main(argv):
    if len(argv) != 2:
        raise SystemExit(
            "usage: bench.py parse_time_point|get_china_candidates|phone_prepare_peak"
        )
    target = argv[1]
    if target == "parse_time_point":
        fn, kind = _parse_time_point()
        result = _measure(fn)
    elif target == "get_china_candidates":
        fn, kind = _get_china_candidates()
        result = _measure(fn)
    elif target == "phone_prepare_peak":
        fn, kind = _phone_prepare_peak()
        result = _measure_peak(fn)
    else:
        raise SystemExit(f"unknown target {target}")
    result["target"] = target
    result["check"] = kind
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv)
