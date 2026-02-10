#!/usr/bin/env python3
"""
Parse llama_memory_breakdown_print output from llama-server log.
Prints one-line summary: GPU (VRAM) and Host (RAM) model/context/compute in MiB.
Usage: parse_memory_breakdown.py [LOG_FILE]
  Reads LOG_FILE or stdin. Looks for "memory breakdown" table lines.
"""
import re
import sys

def main():
    src = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
    lines = src.readlines()
    if src is not sys.stdin:
        src.close()

    # GPU line: "  - CUDA0 (RTX 4090)   | 24077 =  945 + (19187 = 17904 +     384 +     898) +        3945 |"
    # Any device with total = free + (self = model + context + compute) + unaccounted
    gpu_pat = re.compile(
        r"[|\s]+-\s*\S+.*?\|\s*(\d+)\s*=\s*(\d+)\s*\+\s*\((\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\)\s*\+\s*(\d+)"
    )
    # Host line: "  - Host               |                 58271 = 58259 +       0 +      12"
    host_pat = re.compile(
        r"[|\s]+-\s*Host\s+\|\s*(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)"
    )

    gpu_model = gpu_context = gpu_compute = gpu_total = gpu_free = None
    host_model = host_context = host_compute = None

    for line in lines:
        # Strip common log prefix
        s = line.strip()
        if "llama_memory_breakdown_print:" in s:
            s = s.split("llama_memory_breakdown_print:", 1)[1].strip()
        m = gpu_pat.search(s)
        if m:
            gpu_total, gpu_free, _self, gpu_model, gpu_context, gpu_compute, _unacc = (int(x) for x in m.groups())
            continue
        m = host_pat.search(s)
        if m:
            host_self, host_model, host_context, host_compute = (int(x) for x in m.groups())
            continue

    # Output one-line summary for benchmark to show
    parts = []
    if gpu_model is not None:
        parts.append(f"VRAM: model={gpu_model} MiB context={gpu_context} MiB compute={gpu_compute} MiB")
        if gpu_total is not None:
            parts.append(f"total={gpu_total} MiB free={gpu_free} MiB")
    if host_model is not None:
        parts.append(f"RAM: model={host_model} MiB context={host_context} MiB compute={host_compute} MiB")
    if not parts:
        print("(no memory breakdown found in log)")
        return 1
    print("  Memory: " + " | ".join(parts))
    return 0

if __name__ == "__main__":
    sys.exit(main())
