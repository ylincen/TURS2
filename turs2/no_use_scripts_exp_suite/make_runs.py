#!/usr/bin/env python3
import os

# ---- your config ----
RUNNER = "run_all.py"

UCI_SET = [
    "Vehicle", "glass", "pendigits", "HeartCleveland", "chess", "iris",
    "backnote", "contracept", "ionosphere", "car", "tic-tac-toe", "wine",
    "diabetes", "anuran", "avila", "magic", "waveform", "DryBeans",
]

ADBENCH_DIR = "../ADbench_datasets_Classical"
ADBENCH_SET = [
    "26_optdigits.npz", "34_smtp.npz", "28_pendigits.npz", "43_WDBC.npz",
    "36_speech.npz", "31_satimage-2.npz", "3_backdoor.npz", "38_thyroid.npz",
    "41_Waveform.npz", "23_mammography.npz", "40_vowels.npz", "25_musk.npz",
    "1_ALOI.npz", "14_glass.npz",
]

def build_experiments():
    exp1 = {
        "name": "baseline",
        "args": ["--score", "normalized", "--not_use_excl"],
    }
    exp2_aux = {
        "name": "ablation_aux",
        "args": ["--score", "normalized"],
    }
    exp2_nolocal = {
        "name": "ablation_no_local",
        "args": ["--score", "normalized", "--not_use_excl", "--validity_check", "none"],
    }
    exp2_nopatience = {
        "name": "ablation_no_patience",
        "args": ["--score", "normalized", "--not_use_excl", "--no-patience"],
    }
    return [exp1, exp2_aux, exp2_nolocal, exp2_nopatience]


def main():
    out_sh = "run_all_experiments.sh"
    experiments = build_experiments()

    lines = []

    # ---- header + simple parallelism gate ----
    lines.append("#!/usr/bin/env bash")
    lines.append("set -euo pipefail")
    lines.append("")
    lines.append('# Max concurrent jobs; override with env var: MAX_JOBS=16 ./run_all_experiments.sh')
    lines.append('MAX_JOBS="${MAX_JOBS:-32}"')
    lines.append('LOG_DIR="${LOG_DIR:-RIVIUM_OUT_$(date +%Y%m%d)}"')
    lines.append('mkdir -p "$LOG_DIR"')
    lines.append("")
    lines.append('cd "$(dirname "$0")"')
    lines.append("")
    lines.append("wait_for_slot() {")
    lines.append("  while true; do")
    lines.append("    local running")
    lines.append("    running=$(jobs -r -p | wc -l)")
    lines.append("    if (( running < MAX_JOBS )); then")
    lines.append("      break")
    lines.append("    fi")
    lines.append("    sleep 0.2")
    lines.append("  done")
    lines.append("}")
    lines.append("")
    lines.append('PY="${PY:-python}"')
    lines.append("")

    commands = []

    # ---- UCI commands ----
    for exp in experiments:
        for name in UCI_SET:
            label = f"uci_{name}_{exp['name']}"
            cmd = (
                f'"$PY" "{RUNNER}" '
                f'--suite uci --dataset "{name}" '
                + " ".join(exp["args"])
            )
            commands.append((label, cmd))

    # ---- ADBench commands ----
    for exp in experiments:
        for fname in ADBENCH_SET:
            label = f"adbench_{fname}_{exp['name']}".replace(".", "_")
            cmd = (
                f'"$PY" "{RUNNER}" '
                f'--suite adbench --adbench_dir "{ADBENCH_DIR}" --dataset "{fname}" '
                + " ".join(exp["args"])
            )
            commands.append((label, cmd))

    # ---- emit each command as a background job with logging + slot gating ----
    for idx, (label, cmd) in enumerate(commands, start=1):
        log_name = f"{idx:03d}_{label}.log"
        lines.append("wait_for_slot")
        lines.append(f'{{')
        lines.append(f'  echo "[$(date)] START {label}"')
        lines.append(f'  echo "CMD: {cmd}"')
        lines.append(f"  {cmd}")
        lines.append(f'  rc=$?')
        lines.append(f'  echo "[$(date)] END {label} (rc=$rc)"')
        lines.append(f'  exit $rc')
        lines.append(f'}} >"$LOG_DIR/{log_name}" 2>&1 &')
        lines.append("")

    # ---- wait for all jobs ----
    lines.append("echo \"[INFO] Launched ${#commands[@]:-N} jobs (Bash can’t count here, but logs are in $LOG_DIR)\"")
    lines.append("wait")
    lines.append("echo \"[DONE] All background jobs finished (check logs in $LOG_DIR)\"")

    with open(out_sh, "w") as f:
        f.write("\n".join(lines))

    os.chmod(out_sh, 0o755)
    print(f"Wrote {out_sh}")


if __name__ == "__main__":
    main()
