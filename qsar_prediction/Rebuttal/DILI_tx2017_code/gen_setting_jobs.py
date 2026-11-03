"""Generates Phase 2's per-setting SLURM jobs (setting_s<stage>_<i>.slurm)
and submit_tx2017_phase2.sh, using the REAL valid-settings count read from
each stage's results/stage<N>/feature_selection.json -- run this locally
after pulling both feature_selection.json files back from Wahab (Phase 1
must have finished first). Do not guess the count; some AUC-ROC cutoffs may
select 0 features and get filtered out by settings_list.load_settings().

Usage: python gen_setting_jobs.py
"""
import os

from settings_list import load_settings

HERE = os.path.dirname(__file__)
JOBS_DIR = os.path.join(HERE, 'jobs')

SETTING_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=tx17_s{stage}_{idx}
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time={time}
#SBATCH --output=../logs/setting_s{stage}_{idx}_%j.out
#SBATCH --error=../logs/setting_s{stage}_{idx}_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_one_setting.py {stage} {idx}
"""

TIME_BY_STAGE = {1: '00:45:00', 2: '01:00:00'}  # code-variant feature counts go up to 1000
                                                   # (vs. 200 in the paper-text variant), so
                                                   # classifier fits are slower per setting


def main():
    counts = {}
    for stage in (1, 2):
        settings = load_settings(stage)
        n = len(settings)
        counts[stage] = n
        print(f'Stage {stage}: {n} valid settings')
        for i in range(n):
            path = os.path.join(JOBS_DIR, f'setting_s{stage}_{i}.slurm')
            with open(path, 'w') as f:
                f.write(SETTING_TEMPLATE.format(stage=stage, idx=i, time=TIME_BY_STAGE[stage]))
        print(f'  wrote {n} setting_s{stage}_*.slurm files')

    submit_script = f"""#!/bin/bash
# Phase 2 of the two-stage TX2017-Structure pipeline: per-setting jobs for
# both stages, then aggregate_results.py per stage. Requires Phase 1
# (prep + feature_selection.py 1/2) to have already finished, and
# gen_setting_jobs.py to have generated this stage's real setting_s<N>_*.slurm
# files (counts below are what gen_setting_jobs.py found: stage1={counts[1]},
# stage2={counts[2]}).
set -e
cd "$(dirname "$0")/jobs"
mkdir -p ../logs

S1_IDS=()
for i in $(seq 0 {counts[1] - 1}); do
    JID=$(sbatch --parsable "setting_s1_${{i}}.slurm")
    echo "Submitted setting_s1_${{i}}.slurm as job $JID"
    S1_IDS+=($JID)
done
DEP1=$(IFS=:; echo "${{S1_IDS[*]}}")
AGG1_ID=$(sbatch --parsable --dependency=afterok:$DEP1 aggregate_stage1.slurm)
echo "Submitted aggregate_stage1.slurm as job $AGG1_ID (waits for {counts[1]} stage-1 settings)"

S2_IDS=()
for i in $(seq 0 {counts[2] - 1}); do
    JID=$(sbatch --parsable "setting_s2_${{i}}.slurm")
    echo "Submitted setting_s2_${{i}}.slurm as job $JID"
    S2_IDS+=($JID)
done
DEP2=$(IFS=:; echo "${{S2_IDS[*]}}")
AGG2_ID=$(sbatch --parsable --dependency=afterok:$DEP2 aggregate_stage2.slurm)
echo "Submitted aggregate_stage2.slurm as job $AGG2_ID (waits for {counts[2]} stage-2 settings)"

echo
echo "Phase 2 submitted ({counts[1]} + {counts[2]} settings + 2 aggregate = {counts[1] + counts[2] + 2} jobs)."
echo "Check status with: squeue -u \\$USER"
"""
    out_path = os.path.join(HERE, 'submit_tx2017_phase2.sh')
    with open(out_path, 'w') as f:
        f.write(submit_script)
    os.chmod(out_path, 0o755)
    print(f'\nWrote {out_path}')
    print(f'Total Phase 2 jobs: {counts[1]} + {counts[2]} settings + 2 aggregate = {counts[1] + counts[2] + 2}')


if __name__ == '__main__':
    main()
