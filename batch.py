#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from multiprocessing import Process, Queue
from pathlib import Path
from typing import List, Dict, Any


def run_gpu_worker(
    gpu: int, jobs: List[Dict[str, Any]], script_dir: Path, results_queue: Queue
):
    single_sh = script_dir / "single.sh"

    passed = 0
    failed = 0

    for job in jobs:
        print(f"[GPU {gpu}] Starting: {job['model_name']} | {job['backdoor']}")

        cmd = [
            "bash",
            str(single_sh),
            "-mn", job["model_name"],
            "-m", job["model_config"],
            "-d", job["dataset"],
            "-t", job["training"],
            "--wandb", job["wandb"],
            "-g", str(gpu),
            "--output-path", job["output_abs"],
        ]

        if job["backdoor"] and job["backdoor"] != "none":
            cmd.extend(["-bd", job["backdoor"]])

        try:
            subprocess.run(cmd, check=True)
            passed += 1
        except subprocess.CalledProcessError:
            print(
                f"Error: Job failed on GPU {gpu}: {job['model_name']} | {job['backdoor']}",
                file=sys.stderr,
            )
            failed += 1

    results_queue.put((passed, failed))


def resolve_run_dir(target: str, script_dir: Path) -> str:
    if not target:
        print("Error: output path is empty", file=sys.stderr)
        sys.exit(1)

    path = Path(os.path.expanduser(target))
    if path.is_absolute():
        return str(path)
    else:
        return str((script_dir / path).resolve())


def main():
    parser = argparse.ArgumentParser(description="Run batch experiments from JSON.")
    parser.add_argument(
        "experiment_json", help="Path to JSON file in experiments/ (e.g. baseline.json)"
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Print commands only"
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    exp_path = script_dir / "experiments" / args.experiment_json

    if not exp_path.exists():
        print(f"Error: Experiment config not found: {exp_path}", file=sys.stderr)
        sys.exit(1)

    with open(exp_path) as f:
        config = json.load(f)

    gpu_jobs: Dict[int, List[Dict[str, Any]]] = {}

    for group in config:
        gpu = group.get("gpu", 0)
        model_name = group["model_name"]
        model_config = group.get("model", "default.json")
        dataset = group.get("dataset", "default.json")
        training = group.get("training", "default.json")
        wandb = group.get("wandb", "default.json")
        localfs = group.get("localfs", "default.json")
        backdoors = group.get("backdoors", ["none"])
        output_base = group.get("output", "output")

        if gpu not in gpu_jobs:
            gpu_jobs[gpu] = []

        for i, bd in enumerate(backdoors):
            run_output = os.path.join(output_base, f"run{i}")
            output_abs = resolve_run_dir(run_output, script_dir)

            gpu_jobs[gpu].append(
                {
                    "model_name": model_name,
                    "model_config": model_config,
                    "dataset": dataset,
                    "training": training,
                    "wandb": wandb,
                    "localfs": localfs,
                    "backdoor": bd,
                    "output_abs": output_abs,
                }
            )

    print(f"Loaded experiment: {args.experiment_json}")
    print(f"Target GPUs: {sorted(gpu_jobs.keys())}")
    print()

    if args.dry_run:
        single_sh = script_dir / "single.sh"
        for gpu in sorted(gpu_jobs.keys()):
            print(f"=== GPU {gpu} ===")
            for job in gpu_jobs[gpu]:
                cmd = [
                    "./single.sh",
                    "-mn", job["model_name"],
                    "-m", job["model_config"],
                    "-d", job["dataset"],
                    "-t", job["training"],
                    "--wandb", job["wandb"],
                    "-g", str(gpu),
                    "--output-path", job["output_abs"],
                ]
                if job["backdoor"] and job["backdoor"] != "none":
                    cmd.extend(["-bd", job["backdoor"]])
                print(f"  {' '.join(cmd)}")
        return

    results_queue = Queue()
    processes = []

    for gpu, jobs in gpu_jobs.items():
        p = Process(target=run_gpu_worker, args=(gpu, jobs, script_dir, results_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_passed = 0
    total_failed = 0
    while not results_queue.empty():
        p, f = results_queue.get()
        total_passed += p
        total_failed += f

    print()
    print("Batch Summary:")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
