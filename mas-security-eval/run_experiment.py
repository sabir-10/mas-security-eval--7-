#!/usr/bin/env python3
"""Run the full experiment grid and save one JSON line per episode.

Examples
  python run_experiment.py --backend mock --trials 3
  python run_experiment.py --backend anthropic --trials 30 --workers 8 --out results/haiku
  python run_experiment.py --backend gemini --trials 30 --workers 4 --out results/gemini
  python run_experiment.py --backend gemini --rpm 9 --out results/gemini     # free-tier key
  python run_experiment.py --backend ollama:llama3.1:8b --trials 30 --out results/llama
  python run_experiment.py --backend anthropic --dry-run           # just count runs and estimate cost

Re-running the same command resumes: episodes already in results.jsonl are skipped.
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from masec.llm import QuotaExhausted, make_backend
from masec.network import TOPOLOGIES, Defenses, run_episode
from masec.scenarios import ATTACKS


def episode_key(d, topo, attack, trial):
    return f"{d.label}|{topo}|{attack or 'benign'}|{trial}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="mock", help="mock | anthropic[:model] | gemini[:model] | ollama[:model]")
    ap.add_argument("--trials", type=int, default=3, help="repeats of every (defense, attack, layout) cell")
    ap.add_argument("--attacks", default="all", help="comma list, 'all', 'dev', 'heldout' or 'benign' (normal task only)")
    ap.add_argument("--topologies", default="chain,hub")
    ap.add_argument("--defenses", default="all", help="'all' or comma list like none,D1,D1+D2+D3")
    ap.add_argument("--no-benign", action="store_true", help="skip the attack-free benign runs")
    ap.add_argument("--llm-auditor", action="store_true", help="D3 also asks a separate LLM (same backend)")
    ap.add_argument("--taint-tool-results", action="store_true",
                    help="treat tool results as untrusted (a fix to test AFTER the held-out evaluation)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=1, help="parallel episodes (use 4-8 for API backends)")
    ap.add_argument("--rpm", type=float, default=0,
                    help="max model requests per minute (Anthropic/Gemini; use if you hit rate limits)")
    ap.add_argument("--out", default=None, help="results folder (default: results/<backend>)")
    ap.add_argument("--no-traces", action="store_true", help="do not save full message traces")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.attacks == "all":
        attacks = list(ATTACKS)
    elif args.attacks == "benign":            # only the attack-free normal task
        attacks = []
    elif args.attacks in ("dev", "heldout"):
        attacks = [n for n, a in ATTACKS.items() if a.split == args.attacks]
    else:
        attacks = args.attacks.split(",")
    attacks = attacks + ([] if args.no_benign else [None])
    topologies = args.topologies.split(",")
    defenses = Defenses.all_combinations()
    if args.defenses != "all":
        wanted = set(args.defenses.split(","))
        defenses = [d for d in defenses if d.label in wanted]

    args.out = args.out or os.path.join("results", args.backend.replace(":", "_"))
    jobs = [(d, t, a, k) for d in defenses for t in topologies for a in attacks for k in range(args.trials)]
    res_path = os.path.join(args.out, "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        with open(res_path) as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}
    todo = [j for j in jobs if episode_key(*j) not in done]

    calls = len(todo) * (4 + (1 if args.llm_auditor else 0))
    print(f"{len(jobs)} episodes in grid, {len(done)} already done, {len(todo)} to run "
          f"(about {calls} model calls).")
    if args.backend.split(":")[0] in ("anthropic", "gemini"):
        # ~1,500 input + 250 output tokens per call is a rough average for this task.
        print(f"Rough token estimate: {calls*1500/1e6:.1f}M input, {calls*250/1e6:.1f}M output. "
              "Check current prices before running.")
    if args.rpm:
        print(f"Throttled to {args.rpm:g} requests/minute: about {calls / args.rpm / 60:.1f} hours.")
    if args.dry_run or not todo:
        return
    os.makedirs(args.out, exist_ok=True)

    rpm = args.rpm / 2 if args.llm_auditor else args.rpm      # agents and auditor share the limit
    agent_llm = make_backend(args.backend, args.temperature, rpm)
    auditor_llm = make_backend(args.backend, 0.0, rpm) if args.llm_auditor else None
    lock = threading.Lock()
    trace_dir = os.path.join(args.out, "traces")
    os.makedirs(trace_dir, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    def work(job):
        d, t, a, k = job
        r = run_episode(agent_llm, d, t, a, auditor_llm, args.taint_tool_results)
        r.update(key=episode_key(*job), trial=k, backend=args.backend)
        return r

    failures, quota_hit = 0, None
    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(res_path, "a") as out:
        futures = {pool.submit(work, j): j for j in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            if fut.cancelled():
                continue
            try:
                r = fut.result()
            except QuotaExhausted as e:
                quota_hit = str(e)
                for f in futures:
                    f.cancel()           # stop queued episodes; nothing more can succeed today
                continue
            except Exception as e:
                failures += 1
                print(f"  episode {episode_key(*futures[fut])} failed: {e}", file=sys.stderr)
                continue
            trace = r.pop("trace")
            with lock:
                out.write(json.dumps(r) + "\n")
                out.flush()
                if not args.no_traces:
                    with open(os.path.join(trace_dir, r["key"].replace("|", "__") + ".json"), "w") as tf:
                        json.dump(trace, tf, indent=1)
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} episodes done")

    if quota_hit:
        print(f"\nSTOPPED: {quota_hit}", file=sys.stderr)
    u = agent_llm.usage
    if getattr(u, "seconds", 0) and u.calls:
        print(f"Average {u.seconds / u.calls:.1f} s per model call.")
    if u.truncated:
        print(f"WARNING: {u.truncated} replies hit the length limit and were cut off.", file=sys.stderr)
    print(f"Finished. Model calls: {u.calls}, input tokens: {u.input_tokens}, output tokens: {u.output_tokens}."
          + (f" {failures} episodes failed; re-run the same command to retry them." if failures else ""))
    print(f"Results: {res_path}\nNext: python analyze.py {args.out}")


if __name__ == "__main__":
    main()
