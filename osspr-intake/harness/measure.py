"""Baselines for osspr intake. Prints one JSON object as the last stdout line.

Run from the repository root:

    TZ=Asia/Shanghai taskset -c 0 python3 osspr-intake/harness/measure.py word_distribution_peak
    TZ=Asia/Shanghai taskset -c 0 python3 osspr-intake/harness/measure.py idf_peak
    TZ=Asia/Shanghai taskset -c 0 python3 osspr-intake/harness/measure.py pinyin_char_peak
    TZ=Asia/Shanghai taskset -c 0 python3 osspr-intake/harness/measure.py summary_lda_allocs
"""
import ctypes
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import tracemalloc
from pathlib import Path

WARMUP = 2
REPEATS = 9
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _noise_pct(samples):
    """Full spread of the samples, as a percent of the median."""
    median = statistics.median(samples)
    if median == 0:
        return 0.0
    return 100.0 * (max(samples) - min(samples)) / median


def _sha256_lines(lines):
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _word_distribution():
    from jionlp.dictionary.dictionary_loader import word_distribution_loader

    def loaded():
        return word_distribution_loader()

    def fingerprint(result):
        lines = [
            "{}\t{}\t{}\t{}".format(
                word,
                item["total_num"],
                repr(item["prob"]),
                repr(item["log_prob"]),
            )
            for word, item in sorted(result.items())
        ]
        return len(result), _sha256_lines(lines)

    return loaded, fingerprint


def _idf():
    from jionlp.dictionary.dictionary_loader import idf_loader

    def loaded():
        return idf_loader()

    def fingerprint(result):
        lines = [
            "{}\t{}".format(word, repr(value))
            for word, value in sorted(result.items())
        ]
        return len(result), _sha256_lines(lines)

    return loaded, fingerprint


def _pinyin_char():
    from jionlp.dictionary.dictionary_loader import pinyin_char_loader

    def loaded():
        return pinyin_char_loader()

    def fingerprint(result):
        lines = [
            "{}\t{}".format(char, "/".join(pinyin))
            for char, pinyin in sorted(result.items())
        ]
        return len(result), _sha256_lines(lines)

    return loaded, fingerprint


def _summary_lda():
    from jionlp.algorithm.summary.extract_summary import ChineseSummaryExtractor

    def loaded():
        extractor = ChineseSummaryExtractor()
        extractor._lda_prob_matrix()
        return extractor

    def fingerprint(result):
        lines = [
            "{}\t{}".format(word, repr(value))
            for word, value in sorted(result.topic_prominence_dict.items())
        ]
        lines.append("unk\t{}".format(repr(result.unk_topic_prominence_value)))
        lines.append("topics\t{}".format(result.topic_num))
        return len(result.topic_prominence_dict), _sha256_lines(lines)

    return loaded, fingerprint


TARGETS = {
    "word_distribution_peak": ("peak_memory_bytes", _word_distribution),
    "idf_peak": ("peak_memory_bytes", _idf),
    "pinyin_char_peak": ("peak_memory_bytes", _pinyin_char),
    "summary_lda_allocs": ("allocations", _summary_lda),
}


def _measure_peak(fn):
    samples = []
    fingerprint = None
    length = None
    for index in range(WARMUP + REPEATS):
        gc.collect()
        gc.disable()
        tracemalloc.start()
        try:
            result = fn()
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
            gc.enable()
        if index >= WARMUP:
            samples.append(peak)
            length, fingerprint = _remember(result, length, fingerprint)
        del result
        gc.collect()
    return samples, length, fingerprint


def _fingerprint_of(result):
    """Filled in per target by the closure stored on the function."""
    return _fingerprint_of.impl(result)


def _remember(result, length, fingerprint):
    new_length, new_fingerprint = _fingerprint_of(result)
    if fingerprint is None:
        return new_length, new_fingerprint
    if (new_length, new_fingerprint) != (length, fingerprint):
        raise SystemExit("observable result changed between repeats")
    return length, fingerprint


def _libpython():
    names = []
    for key in ("INSTSONAME", "LDLIBRARY", "LIBRARY"):
        value = sysconfig.get_config_var(key)
        if value and not str(value).endswith(".a"):
            names.append(value)
    names.extend([
        "libpython{}.so.1.0".format(sysconfig.get_config_var("LDVERSION") or ""),
        "libpython{}.so.1".format(sysconfig.get_config_var("LDVERSION") or ""),
        "libpython{}.so".format(sysconfig.get_config_var("LDVERSION") or ""),
    ])
    directories = []
    for key in ("LIBDIR", "LIBPL"):
        value = sysconfig.get_config_var(key)
        if value:
            directories.append(value)
    directories.extend([
        "/usr/lib/x86_64-linux-gnu",
        "/lib/x86_64-linux-gnu",
    ])
    seen = set()
    for directory in directories:
        for name in names:
            if not name or name in seen:
                continue
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
    raise SystemExit("libpython shared library not found")


def _alloc_library():
    source = Path(__file__).resolve().parent / "alloc_count.c"
    built = Path(tempfile.gettempdir()) / "ossp-intake-alloc-count.so"
    if (not built.exists()) or built.stat().st_mtime < source.stat().st_mtime:
        subprocess.check_call(
            [
                "gcc", "-shared", "-fPIC", "-O2",
                "-o", str(built), str(source), _libpython(),
            ],
            stdout=subprocess.DEVNULL,
        )
    library = ctypes.CDLL(str(built))
    library.alloc_count_install.restype = None
    library.alloc_count_uninstall.restype = ctypes.c_uint64
    return library


def _measure_allocs(fn):
    library = _alloc_library()
    samples = []
    fingerprint = None
    length = None
    for index in range(WARMUP + REPEATS):
        gc.collect()
        gc.disable()
        library.alloc_count_install()
        try:
            result = fn()
            count = int(library.alloc_count_uninstall())
        finally:
            library.alloc_count_uninstall()
            gc.enable()
        if index >= WARMUP:
            samples.append(count)
            length, fingerprint = _remember(result, length, fingerprint)
        del result
        gc.collect()
    return samples, length, fingerprint


def main(argv):
    if len(argv) != 2 or argv[1] not in TARGETS:
        names = "|".join(TARGETS)
        raise SystemExit("usage: measure.py {}".format(names))
    target = argv[1]
    metric_kind, factory = TARGETS[target]
    fn, fingerprint = factory()
    _fingerprint_of.impl = fingerprint
    if metric_kind == "peak_memory_bytes":
        samples, length, digest = _measure_peak(fn)
        value_key = "median_bytes"
    else:
        samples, length, digest = _measure_allocs(fn)
        value_key = "median_allocations"
    median = statistics.median(samples)
    payload = {
        "target": target,
        "metric_kind": metric_kind,
        value_key: median,
        "noise_pct": _noise_pct(samples),
        "samples": samples,
        "result_len": length,
        "result_sha256": digest,
    }
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv)
