#!/usr/bin/env python3
"""
Run Qwen3-Coder-Next (and related) benchmark: start server → measure tok/s → stop.
Replaces the logic of benchmark.sh; invoke via benchmark.sh for conda/env wrapper.
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
PORT = 18999
RESULTS_FILE = BENCH_DIR / "results.txt"
SERVER_LOG = BENCH_DIR / ".server_log.txt"


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run benchmark: start llama-server per scenario, measure short/long tok/s, write results."
    )
    ap.add_argument(
        "--long",
        action="store_true",
        help="Run both short and long-context tests (default: short only)",
    )
    ap.add_argument(
        "--short-only",
        action="store_true",
        help="Run short-context only (default)",
    )
    ap.add_argument(
        "--ctx",
        metavar="SIZE",
        help="Context size override, e.g. 128k or 131072 (server -c)",
    )
    ap.add_argument(
        "--no-cpu",
        action="store_true",
        help="Skip CPU (system-only) pass",
    )
    ap.add_argument(
        "--long-chars",
        type=int,
        default=100000,
        metavar="N",
        help="Max character length for long-context prompt (default: 100000, ~25k tokens est.). If resulting tokens exceed server -c (see --ctx), the long-context request will fail.",
    )
    ap.add_argument(
        "scenario",
        nargs="?",
        help="Run only this scenario (e.g. mxfp4_full)",
    )
    args = ap.parse_args()
    # Resolve context size
    ctx_val = None
    if args.ctx:
        if args.ctx.lower().endswith("k"):
            try:
                ctx_val = str(int(args.ctx[:-1]) * 1024)
            except ValueError:
                ap.error(f"Invalid --ctx: {args.ctx!r} (use e.g. 128k or 131072)")
        elif args.ctx.isdigit():
            ctx_val = args.ctx
        else:
            ap.error(f"Invalid --ctx: {args.ctx!r} (use e.g. 128k or 131072)")
    return args, ctx_val


def load_scenarios(only_scenario: str | None) -> list[str]:
    cfg = BENCH_DIR / "scenarios.cfg"
    names = []
    with open(cfg) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = line.split("|")[0].strip()
            if only_scenario:
                if name == only_scenario:
                    return [name]
            else:
                names.append(name)
    if only_scenario:
        known = " ".join(load_scenarios(None))
        print(f"Unknown scenario: {only_scenario}", file=sys.stderr)
        print(f"Known: {known}", file=sys.stderr)
        sys.exit(1)
    return names


def get_api_model(scenario: str) -> str:
    cfg = BENCH_DIR / "scenarios.cfg"
    with open(cfg) as f:
        for line in f:
            if not line.strip() or line.strip().startswith("#"):
                continue
            parts = [p.strip() for p in line.strip().split("|")]
            if parts and parts[0] == scenario and len(parts) >= 6 and parts[5]:
                return parts[5]
    return "qwen3-coder-next"


def wait_for_models(base_url: str, timeout_s: int = 60) -> bool:
    for i in range(timeout_s):
        try:
            r = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", f"{base_url}/v1/models"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip() == "200":
                return True
        except Exception:
            pass
        if i > 0 and i % 10 == 0:
            print(f"  ... waiting for server ({i}s)", flush=True)
        time.sleep(1)
    return False


def wait_for_model_ready(base_url: str, api_model: str, timeout_attempts: int = 60) -> tuple[bool, str]:
    for attempt in range(timeout_attempts):
        try:
            r = subprocess.run(
                [
                    "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                    "-X", "POST", f"{base_url}/v1/chat/completions",
                    "-H", "Content-Type: application/json",
                    "-d", f'{{"model":"{api_model}","messages":[{{"role":"user","content":"x"}}],"max_tokens":1}}',
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            code = (r.stdout or "").strip() or "000"
            if code == "200":
                return True, code
        except Exception:
            code = "000"
        if attempt > 0 and attempt % 3 == 0:
            print(f"  ... still loading ({attempt * 5}s)", flush=True)
        time.sleep(5)
    return False, code


def run_measure(port: int, api_model: str, prompt_file: Path | None = None) -> str:
    cmd = [sys.executable, str(BENCH_DIR / "measure.py"), "--port", str(port), "--model", api_model]
    if prompt_file and prompt_file.exists():
        cmd.extend(["--prompt-file", str(prompt_file)])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(BENCH_DIR))
    return (r.stdout or "") + (r.stderr or "")


def parse_measure_out(out: str) -> tuple[str | None, str | None, str | None]:
    tok_s = None
    ctx = None
    time_s = None
    if not out:
        return tok_s, ctx, time_s
    m = re.search(r"tok/s=([0-9.]+)", out)
    if m:
        tok_s = m.group(1)
    m = re.search(r"gen_tok_s=([0-9.]+)", out)
    if m:
        tok_s = m.group(1)
    m = re.search(r"prompt_tokens=([0-9]+)", out)
    if m:
        ctx = m.group(1)
    m = re.search(r"time=([0-9.]+)s", out)
    if m:
        time_s = m.group(1) + "s"
    return tok_s, ctx, time_s


def run_parse_memory(log_path: Path) -> str | None:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return None
    r = subprocess.run(
        [sys.executable, str(BENCH_DIR / "parse_memory_breakdown.py"), str(log_path)],
        capture_output=True,
        text=True,
        timeout=5,
        cwd=str(BENCH_DIR),
    )
    line = (r.stdout or "").strip()
    if line and "no memory breakdown" not in line:
        return line
    return None


def run_pass(
    scenarios: list[str],
    short_only: bool,
    long_chars: int,
    pass_env: dict,
    backend_label: str,
    results_lines: list[str],
) -> None:
    base_url = f"http://127.0.0.1:{PORT}"
    fill_path = BENCH_DIR / ".long_prompt.txt"

    for scenario in scenarios:
        print(f"--- {scenario} ({backend_label}) ---")
        api_model = get_api_model(scenario)

        # Clear and start server (keep log handle open so server output is captured)
        with open(SERVER_LOG, "w"):
            pass
        log_file = open(SERVER_LOG, "a")
        env = os.environ.copy()
        env.update(pass_env)
        proc = subprocess.Popen(
            [str(BENCH_DIR / "run_server.sh"), scenario, str(PORT)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(BENCH_DIR),
            start_new_session=True,
        )
        try:
            if not wait_for_models(base_url):
                print("Server did not become ready for", scenario)
                proc.terminate()
                proc.wait(timeout=10)
                results_lines.append((scenario, "FAIL", "-", "-", "-", "-", "-"))
                with open(RESULTS_FILE, "a") as f:
                    f.write(f"{scenario:<18} {'FAIL':>12} {'-':>12} {'-':>12} {'-':>12} {'-':>8} {'-':>8}\n")
                log_file.close()
                continue

            print("  Waiting for model to load...")
            ready, code = wait_for_model_ready(base_url, api_model)
            if not ready:
                print(f"  Model did not become ready (got HTTP {code}); skipping measure")
                proc.terminate()
                proc.wait(timeout=10)
                results_lines.append((scenario, "FAIL", "-", "-", "-", "-", "-"))
                with open(RESULTS_FILE, "a") as f:
                    f.write(f"{scenario:<18} {'FAIL':>12} {'-':>12} {'-':>12} {'-':>12} {'-':>8} {'-':>8}\n")
                log_file.close()
                continue

            # Short measure
            out_short = run_measure(PORT, api_model, None)
            tok_s_short, short_ctx, short_time = parse_measure_out(out_short)
            if not tok_s_short and out_short:
                print("  measure:", out_short[:200])

            tok_s_long: str | None = None
            long_ctx: str | None = None
            long_time: str | None = None
            if not short_only:
                # Generate long prompt (capped at long_chars characters)
                with open(fill_path, "w") as f:
                    subprocess.run(
                        [str(BENCH_DIR / "fill_context.sh"), str(long_chars)],
                        stdout=f,
                        stderr=subprocess.DEVNULL,
                        timeout=60,
                        cwd=str(BENCH_DIR),
                    )
                if fill_path.exists() and fill_path.stat().st_size > 0:
                    long_prompt_chars = len(fill_path.read_text())
                    print(f"  Long prompt: {long_prompt_chars} chars (~{long_prompt_chars // 4} tokens est.)", end="", flush=True)
                    if long_prompt_chars < long_chars:
                        print(f" (requested {long_chars}; limited by project files in proxy/ + stack/)", flush=True)
                    else:
                        print(flush=True)
                    out_long = run_measure(PORT, api_model, fill_path)
                    tok_s_long, long_ctx, long_time = parse_measure_out(out_long)
                    if long_ctx:
                        print(f"  Long prompt: {long_prompt_chars} chars, {long_ctx} tokens (actual)", flush=True)
                    if not tok_s_long and out_long:
                        print("  measure (long) failed or no parseable output:", out_long.strip()[:500], flush=True)

            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

            # Memory breakdown
            mem_line = run_parse_memory(SERVER_LOG)
            if mem_line:
                print(mem_line)
                with open(RESULTS_FILE, "a") as f:
                    f.write(mem_line + "\n")

            short_s = tok_s_short if tok_s_short else "-"
            long_s = tok_s_long if tok_s_long else "-"
            sc = short_ctx if short_ctx else "-"
            lc = long_ctx if long_ctx else "-"
            st = short_time if short_time else "-"
            lt = long_time if long_time else "-"
            results_lines.append((scenario, short_s, long_s, sc, lc, st, lt))
            with open(RESULTS_FILE, "a") as f:
                f.write(f"{scenario:<18} {short_s:>12} {long_s:>12} {sc:>12} {lc:>12} {st:>8} {lt:>8}\n")
            print(f"  short: {short_s} tok/s ctx={sc} time={st}  long: {long_s} gen/s ctx={lc} time={lt}")

        except KeyboardInterrupt:
            proc.terminate()
            proc.wait(timeout=5)
            raise
        finally:
            try:
                log_file.close()
            except Exception:
                pass
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)


def main() -> int:
    args, ctx_val = parse_args()
    short_only = not args.long or args.short_only
    run_cpu = not args.no_cpu and os.environ.get("RUN_CPU", "1") == "1"
    only_scenario = args.scenario

    scenarios = load_scenarios(only_scenario)

    if only_scenario:
        print("Mode: single scenario only:", only_scenario)
    elif short_only:
        print("Mode: short only (use --long to also run long-context tests)")
    else:
        print("Mode: short + long context (--long)")
    if run_cpu and not only_scenario:
        print("Will run GPU then CPU (system-only) pass for comparison")
    if ctx_val:
        print("Context override:", ctx_val, "tokens (server -c; model max may be higher)")
    if not short_only:
        ctx_limit = int(ctx_val) if ctx_val else 32768
        est_tokens = args.long_chars // 4
        print("Long prompt: max", args.long_chars, "chars (--long-chars), ~", est_tokens, "tokens est.")
        if est_tokens > ctx_limit:
            print("  Warning: estimated tokens (~", est_tokens, ") > server context (", ctx_limit, "). Long-context request may fail.", sep="")
    print()

    pass_env = {}
    if ctx_val:
        pass_env["BENCHMARK_CTX"] = ctx_val

    # Init results file
    try:
        date_str = subprocess.run(
                            ["date", "-Iseconds"],
                            capture_output=True,
                            text=True,
                            timeout=1,
                        ).stdout.strip()
    except Exception:
        date_str = time.strftime("%Y-%m-%d %H:%M:%S")
    mode_str = f"single scenario: {only_scenario}" if only_scenario else ("short only" if short_only else "short + long (--long)")
    with open(RESULTS_FILE, "w") as f:
        f.write(f"Qwen3-Coder-Next benchmark — {date_str}\n")
        f.write(f"Mode: {mode_str}\n\n")
        f.write("=== GPU ===\n")
        f.write(f"{'Scenario':<18} {'Short tok/s':>12} {'Long gen/s':>12} {'Short ctx':>12} {'Long ctx':>12} {'Short t':>8} {'Long t':>8}\n")
        f.write(f"{'-------':<18} {'----------':>12} {'----------':>12} {'---------':>12} {'--------':>12} {'------':>8} {'------':>8}\n")

    long_chars = args.long_chars
    results_gpu: list[tuple[str, str, str, str, str, str, str]] = []
    run_pass(scenarios, short_only, long_chars, pass_env, "GPU", results_gpu)

    if run_cpu and not only_scenario:
        print("\n=== CPU (system only) pass ===")
        with open(RESULTS_FILE, "a") as f:
            f.write("\n=== CPU (system only) ===\n")
            f.write(f"{'Scenario':<18} {'Short tok/s':>12} {'Long gen/s':>12} {'Short ctx':>12} {'Long ctx':>12} {'Short t':>8} {'Long t':>8}\n")
            f.write(f"{'-------':<18} {'----------':>12} {'----------':>12} {'---------':>12} {'--------':>12} {'------':>8} {'------':>8}\n")
        cpu_env = {**pass_env, "N_GPU_LAYERS": "0"}
        run_pass(scenarios, short_only, long_chars, cpu_env, "CPU", [])

    print("\n=== Summary ===")
    with open(RESULTS_FILE) as f:
        print(f.read())
    print("Results written to", RESULTS_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
