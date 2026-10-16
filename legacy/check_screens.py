#!/usr/bin/env python3

import os
import re
import shutil
import subprocess
import argparse
from collections import defaultdict

IDLE_NAMES = {
    "screen", "bash", "zsh", "sh", "dash", "fish",
    "vim", "nvim", "nano", "emacs",
    "less", "more", "man", "cat",
    "top", "htop", "watch",
    "ssh", "sshd",
    "tail", "sleep",
}

ACTIVE_HINTS = {
    "python", "python3", "ipython", "jupyter",
    "rscript", "r", "julia",
    "matlab",
    "torchrun", "accelerate",
    "mpirun", "mpiexec",
    "snakemake", "nextflow",
    "java", "node", "ruby", "perl",
    "singularity", "apptainer", "docker",
}


def run(cmd):
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def parse_screen_ls():
    result = run(["screen", "-ls"])
    sessions = []

    pattern = re.compile(
        r"^\s*(\d+\.[^\s]+)\s+\(([^)]+)\)\s+\((Attached|Detached)\)"
    )

    for line in result.stdout.splitlines():
        match = pattern.match(line)
        if match:
            name = match.group(1)
            created = match.group(2)
            state = match.group(3)
            pid = int(name.split(".", 1)[0])
            sessions.append(
                {
                    "name": name,
                    "pid": pid,
                    "created": created,
                    "state": state,
                }
            )

    return sessions


def build_children_map():
    children = defaultdict(list)

    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue

        pid = int(entry)

        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                        children[ppid].append(pid)
                        break
        except FileNotFoundError:
            continue
        except PermissionError:
            continue

    return children


def descendants(pid, children_map):
    found = []
    stack = list(children_map.get(pid, []))

    while stack:
        child = stack.pop()
        found.append(child)
        stack.extend(children_map.get(child, []))

    return found


def get_ps_info(pids):
    if not pids:
        return {}

    result = run([
        "ps",
        "-o", "pid=,ppid=,stat=,etime=,%cpu=,%mem=,args=",
        "-p", ",".join(str(p) for p in sorted(set(pids))),
    ])

    info = {}

    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 6)
        if len(parts) < 7:
            continue

        pid, ppid, stat, etime, cpu, mem, args = parts
        pid = int(pid)

        info[pid] = {
            "pid": pid,
            "ppid": int(ppid),
            "stat": stat,
            "etime": etime,
            "cpu": cpu,
            "mem": mem,
            "args": args,
            "cmd": os.path.basename(args.split()[0]) if args else "",
        }

    return info


def looks_active(proc):
    args = proc["args"].lower()
    cmd = proc["cmd"].lower()

    if not cmd:
        return False

    if cmd in ACTIVE_HINTS:
        return True

    if any(hint in args for hint in ACTIVE_HINTS):
        return True

    if ".py" in args or ".r" in args or ".jl" in args or ".sh" in args:
        return True

    if cmd in IDLE_NAMES:
        return False

    return True


def get_gpu_pids():
    if not shutil.which("nvidia-smi"):
        return {}

    result = run([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])

    gpu = {}

    for line in result.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            gpu[int(parts[0])] = {
                "process": parts[1],
                "memory_mb": parts[2],
            }

    return gpu


def short_cmd(cmd, width=90):
    return cmd if len(cmd) <= width else cmd[: width - 3] + "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--details",
        action="store_true",
        help="show all descendant processes, not only likely active ones",
    )
    args = parser.parse_args()

    sessions = parse_screen_ls()

    if not sessions:
        print("No screen sessions found.")
        return

    children_map = build_children_map()

    all_relevant_pids = []
    session_descendants = {}

    for session in sessions:
        pids = descendants(session["pid"], children_map)
        session_descendants[session["pid"]] = pids
        all_relevant_pids.extend([session["pid"]] + pids)

    ps_info = get_ps_info(all_relevant_pids)
    gpu_pids = get_gpu_pids()

    print()
    print("Screen session report")
    print("=" * 120)
    print(
        f"{'SESSION':38} {'STATE':9} {'VERDICT':14} {'GPU':5} {'ACTIVE / IMPORTANT COMMAND'}"
    )
    print("-" * 120)

    reports = []

    for session in sessions:
        screen_pid = session["pid"]
        child_pids = session_descendants.get(screen_pid, [])
        child_infos = [
            ps_info[p] for p in child_pids
            if p in ps_info
        ]

        active = [p for p in child_infos if looks_active(p)]
        gpu_hits = [p for p in child_pids if p in gpu_pids]

        if gpu_hits:
            verdict = "ACTIVE-GPU"
        elif active:
            verdict = "ACTIVE?"
        elif child_infos:
            verdict = "probably idle"
        else:
            verdict = "empty/dead?"

        gpu_flag = "yes" if gpu_hits else "no"

        if active:
            main_cmd = short_cmd(active[0]["args"])
        elif child_infos:
            main_cmd = short_cmd(child_infos[0]["args"])
        else:
            main_cmd = ""

        print(
            f"{session['name'][:37]:38} "
            f"{session['state'][:8]:9} "
            f"{verdict[:13]:14} "
            f"{gpu_flag:5} "
            f"{main_cmd}"
        )

        reports.append((session, active, child_infos, gpu_hits))

    print("-" * 120)

    print()
    print("Details")
    print("=" * 120)

    for session, active, child_infos, gpu_hits in reports:
        print()
        print(f"{session['name']}  [{session['state']}]  created {session['created']}")

        if gpu_hits:
            print("  GPU processes:")
            for pid in gpu_hits:
                g = gpu_pids[pid]
                proc = ps_info.get(pid, {})
                print(
                    f"    PID {pid} | GPU mem {g['memory_mb']} MB | "
                    f"{short_cmd(proc.get('args', g['process']))}"
                )

        rows = child_infos if args.details else active

        if not rows:
            if child_infos:
                print("  No obvious code process found. Only shell/idle-looking processes.")
                if not args.details:
                    print("  Run with --details to see them.")
            else:
                print("  No child processes found.")
            continue

        print("  Processes:")
        print("    PID       STAT   ELAPSED      CPU%   MEM%   COMMAND")
        for p in rows:
            print(
                f"    {p['pid']:<9} "
                f"{p['stat']:<6} "
                f"{p['etime']:<12} "
                f"{p['cpu']:<6} "
                f"{p['mem']:<6} "
                f"{short_cmd(p['args'])}"
            )

    print()
    print("Legend:")
    print("  ACTIVE-GPU     a descendant process is using the GPU")
    print("  ACTIVE?        a likely code/job process was found")
    print("  probably idle  only shell or idle-looking processes were found")
    print()
    print("This script does not attach to or modify any screen session.")


if __name__ == "__main__":
    main()