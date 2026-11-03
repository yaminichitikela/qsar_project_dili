"""Generates every SLURM job file (GNN family x4 archs, GT family x2 archs,
classical x3 models x2 stages, consensus) plus submit_all_cv.sh, for the
full 10-model DILI CV pipeline. MV-GNN itself already has its own jobs/
submit_dili_cv.sh from the earlier build and is NOT regenerated here.

Run this once locally, then upload jobs/ + submit_all_cv.sh to Wahab.

Usage: python gen_all_cv_jobs.py
"""
import os

HERE = os.path.dirname(__file__)
JOBS_DIR = os.path.join(HERE, 'jobs')
os.makedirs(JOBS_DIR, exist_ok=True)

GNN_ARCHS = ['GCN', 'GAT', 'GIN', 'SAGE']
GT_ARCHS = ['Graphormer', 'GraphGPS']
CLASSICAL_MODELS = ['RF', 'SVM', 'XGB']
N_GNN_CONFIGS = 3   # len(GNN_CONFIGS[arch]) for every arch
N_GT_CONFIGS = 3    # len(GT_CONFIGS[arch]) for every arch

HEADER_GPU = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=odu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time={time}
#SBATCH --output=../logs/{log_name}_%j.out
#SBATCH --error=../logs/{log_name}_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python {cmd}
"""

HEADER_CPU = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output=../logs/{log_name}_%j.out
#SBATCH --error=../logs/{log_name}_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python {cmd}
"""


def write_gpu(fname, job_name, log_name, cmd, time):
    with open(os.path.join(JOBS_DIR, fname), 'w') as f:
        f.write(HEADER_GPU.format(job_name=job_name, log_name=log_name, cmd=cmd, time=time))


def write_cpu(fname, job_name, log_name, cmd, time, cpus=4, mem='8G'):
    with open(os.path.join(JOBS_DIR, fname), 'w') as f:
        f.write(HEADER_CPU.format(job_name=job_name, log_name=log_name, cmd=cmd, time=time, cpus=cpus, mem=mem))


def main():
    n_files = 0

    # ── GNN family (GCN/GAT/GIN/SAGE) ───────────────────────────────────────
    for arch in GNN_ARCHS:
        a = arch.lower()
        for i in range(N_GNN_CONFIGS):
            write_gpu(f'gnn_{a}_cfg{i}.slurm', f'g_{a}_c{i}', f'gnn_{a}_cfg{i}',
                      f'train_gnn_cv_config.py {arch} {i}', '12:00:00')
            n_files += 1
        write_cpu(f'gnn_{a}_select.slurm', f'g_{a}_sel', f'gnn_{a}_select',
                   f'select_gnn_stage1_config.py {arch}', '00:15:00', cpus=2)
        n_files += 1
        for i in range(5):
            write_gpu(f'gnn_{a}_s1fold{i}.slurm', f'g_{a}_1f{i}', f'gnn_{a}_s1fold{i}',
                       f'train_gnn_stage1_fold.py {arch} {i}', '04:00:00')
            n_files += 1
        write_cpu(f'gnn_{a}_agg1.slurm', f'g_{a}_a1', f'gnn_{a}_agg1',
                   f'aggregate_gnn_stage1_cv.py {arch}', '00:15:00', cpus=2)
        n_files += 1
        for i in range(5):
            write_gpu(f'gnn_{a}_s2fold{i}.slurm', f'g_{a}_2f{i}', f'gnn_{a}_s2fold{i}',
                       f'train_gnn_stage2_fold.py {arch} {i}', '03:00:00')
            n_files += 1
        write_cpu(f'gnn_{a}_agg2.slurm', f'g_{a}_a2', f'gnn_{a}_agg2',
                   f'aggregate_gnn_stage2_cv.py {arch}', '00:15:00', cpus=2)
        n_files += 1

    # ── GT family (Graphormer/GraphGPS) ─────────────────────────────────────
    for arch in GT_ARCHS:
        a = arch.lower()
        for i in range(N_GT_CONFIGS):
            write_gpu(f'gt_{a}_cfg{i}.slurm', f't_{a}_c{i}', f'gt_{a}_cfg{i}',
                       f'train_gt_cv_config.py {arch} {i}', '14:00:00')
            n_files += 1
        write_cpu(f'gt_{a}_select.slurm', f't_{a}_sel', f'gt_{a}_select',
                   f'select_gt_stage1_config.py {arch}', '00:15:00', cpus=2)
        n_files += 1
        for i in range(5):
            write_gpu(f'gt_{a}_s1fold{i}.slurm', f't_{a}_1f{i}', f'gt_{a}_s1fold{i}',
                       f'train_gt_stage1_fold.py {arch} {i}', '05:00:00')
            n_files += 1
        write_cpu(f'gt_{a}_agg1.slurm', f't_{a}_a1', f'gt_{a}_agg1',
                   f'aggregate_gt_stage1_cv.py {arch}', '00:15:00', cpus=2)
        n_files += 1
        for i in range(5):
            write_gpu(f'gt_{a}_s2fold{i}.slurm', f't_{a}_2f{i}', f'gt_{a}_s2fold{i}',
                       f'train_gt_stage2_fold.py {arch} {i}', '04:00:00')
            n_files += 1
        write_cpu(f'gt_{a}_agg2.slurm', f't_{a}_a2', f'gt_{a}_agg2',
                   f'aggregate_gt_stage2_cv.py {arch}', '00:15:00', cpus=2)
        n_files += 1

    # ── Classical (RF/SVM/XGB) -- one job per (model, stage), CPU-only ──────
    for model in CLASSICAL_MODELS:
        m = model.lower()
        for stage in (1, 2):
            write_cpu(f'classical_{m}_s{stage}.slurm', f'c_{m}_s{stage}', f'classical_{m}_s{stage}',
                       f'train_classical_cv.py {model} {stage}', '02:00:00', cpus=8, mem='16G')
            n_files += 1

    # ── Consensus ─────────────────────────────────────────────────────────
    write_cpu('consensus_cv.slurm', 'cv_consensus', 'consensus_cv',
               'compute_consensus_cv.py', '00:15:00', cpus=2)
    n_files += 1

    print(f'Wrote {n_files} SLURM job files to {JOBS_DIR}/')

    # ── submit script ────────────────────────────────────────────────────
    lines = []
    lines.append('#!/bin/bash')
    lines.append('# Submits the full 10-model DILI CV pipeline (GNN family x4, GT family x2,')
    lines.append('# classical x3, consensus). Requires prep_dili_cv.slurm (rewritten to build')
    lines.append('# ECFP4 + gt-graph representations too) to have already finished --')
    lines.append('# submit_dili_cv.sh (MV-GNN) already runs it; run that first, or run')
    lines.append('# prep_dili_cv.slurm standalone if MV-GNN was already done under the old prep.')
    lines.append('set -e')
    lines.append('cd "$(dirname "$0")/jobs"')
    lines.append('mkdir -p ../logs')
    lines.append('')
    lines.append('ALL_DEP_IDS=()')
    lines.append('')

    for arch in GNN_ARCHS:
        a = arch.lower()
        lines.append(f'# --- GNN: {arch} ---')
        lines.append(f'{a.upper()}_CFG_IDS=()')
        for i in range(N_GNN_CONFIGS):
            lines.append(f'JID=$(sbatch --parsable "gnn_{a}_cfg{i}.slurm"); echo "Submitted gnn_{a}_cfg{i}.slurm as job $JID"; {a.upper()}_CFG_IDS+=($JID)')
        lines.append(f'DEP=$(IFS=:; echo "${{{a.upper()}_CFG_IDS[*]}}")')
        lines.append(f'{a.upper()}_SEL_ID=$(sbatch --parsable --dependency=afterok:$DEP "gnn_{a}_select.slurm"); echo "Submitted gnn_{a}_select.slurm as job ${a.upper()}_SEL_ID"')
        lines.append(f'{a.upper()}_S1_IDS=()')
        for i in range(5):
            lines.append(f'JID=$(sbatch --parsable --dependency=afterok:${a.upper()}_SEL_ID "gnn_{a}_s1fold{i}.slurm"); echo "Submitted gnn_{a}_s1fold{i}.slurm as job $JID"; {a.upper()}_S1_IDS+=($JID)')
        lines.append(f'DEP=$(IFS=:; echo "${{{a.upper()}_S1_IDS[*]}}")')
        lines.append(f'{a.upper()}_AGG1_ID=$(sbatch --parsable --dependency=afterok:$DEP "gnn_{a}_agg1.slurm"); echo "Submitted gnn_{a}_agg1.slurm as job ${a.upper()}_AGG1_ID"')
        lines.append(f'{a.upper()}_S2_IDS=()')
        for i in range(5):
            lines.append(f'JID=$(sbatch --parsable --dependency=afterok:${a.upper()}_AGG1_ID "gnn_{a}_s2fold{i}.slurm"); echo "Submitted gnn_{a}_s2fold{i}.slurm as job $JID"; {a.upper()}_S2_IDS+=($JID)')
        lines.append(f'DEP=$(IFS=:; echo "${{{a.upper()}_S2_IDS[*]}}")')
        lines.append(f'{a.upper()}_AGG2_ID=$(sbatch --parsable --dependency=afterok:$DEP "gnn_{a}_agg2.slurm"); echo "Submitted gnn_{a}_agg2.slurm as job ${a.upper()}_AGG2_ID"')
        lines.append(f'ALL_DEP_IDS+=(${a.upper()}_AGG1_ID ${a.upper()}_AGG2_ID)')
        lines.append('')

    for arch in GT_ARCHS:
        a = arch.lower()
        v = a.upper()
        lines.append(f'# --- GT: {arch} ---')
        lines.append(f'{v}_CFG_IDS=()')
        for i in range(N_GT_CONFIGS):
            lines.append(f'JID=$(sbatch --parsable "gt_{a}_cfg{i}.slurm"); echo "Submitted gt_{a}_cfg{i}.slurm as job $JID"; {v}_CFG_IDS+=($JID)')
        lines.append(f'DEP=$(IFS=:; echo "${{{v}_CFG_IDS[*]}}")')
        lines.append(f'{v}_SEL_ID=$(sbatch --parsable --dependency=afterok:$DEP "gt_{a}_select.slurm"); echo "Submitted gt_{a}_select.slurm as job ${v}_SEL_ID"')
        lines.append(f'{v}_S1_IDS=()')
        for i in range(5):
            lines.append(f'JID=$(sbatch --parsable --dependency=afterok:${v}_SEL_ID "gt_{a}_s1fold{i}.slurm"); echo "Submitted gt_{a}_s1fold{i}.slurm as job $JID"; {v}_S1_IDS+=($JID)')
        lines.append(f'DEP=$(IFS=:; echo "${{{v}_S1_IDS[*]}}")')
        lines.append(f'{v}_AGG1_ID=$(sbatch --parsable --dependency=afterok:$DEP "gt_{a}_agg1.slurm"); echo "Submitted gt_{a}_agg1.slurm as job ${v}_AGG1_ID"')
        lines.append(f'{v}_S2_IDS=()')
        for i in range(5):
            lines.append(f'JID=$(sbatch --parsable --dependency=afterok:${v}_AGG1_ID "gt_{a}_s2fold{i}.slurm"); echo "Submitted gt_{a}_s2fold{i}.slurm as job $JID"; {v}_S2_IDS+=($JID)')
        lines.append(f'DEP=$(IFS=:; echo "${{{v}_S2_IDS[*]}}")')
        lines.append(f'{v}_AGG2_ID=$(sbatch --parsable --dependency=afterok:$DEP "gt_{a}_agg2.slurm"); echo "Submitted gt_{a}_agg2.slurm as job ${v}_AGG2_ID"')
        lines.append(f'ALL_DEP_IDS+=(${v}_AGG1_ID ${v}_AGG2_ID)')
        lines.append('')

    lines.append('# --- Classical: RF/SVM/XGB, both stages (independent, CPU-only) ---')
    for model in CLASSICAL_MODELS:
        m = model.lower()
        for stage in (1, 2):
            var = f'{m.upper()}_S{stage}_ID'
            lines.append(f'{var}=$(sbatch --parsable "classical_{m}_s{stage}.slurm"); echo "Submitted classical_{m}_s{stage}.slurm as job ${var}"')
            lines.append(f'ALL_DEP_IDS+=(${var})')
    lines.append('')

    lines.append('# --- Consensus: waits for every model above (MV-GNN already has its own')
    lines.append('#     completed results from the earlier submit_dili_cv.sh run) ---')
    lines.append('DEP=$(IFS=:; echo "${ALL_DEP_IDS[*]}")')
    lines.append('CONS_ID=$(sbatch --parsable --dependency=afterok:$DEP consensus_cv.slurm)')
    lines.append('echo "Submitted consensus_cv.slurm as job $CONS_ID (waits for all 9 non-MVGNN models: $DEP)"')
    lines.append('')
    lines.append('echo')
    lines.append('echo "All CV jobs submitted."')
    lines.append('echo "Check status with: squeue -u \\$USER"')

    submit_path = os.path.join(HERE, 'submit_all_cv.sh')
    with open(submit_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    os.chmod(submit_path, 0o755)
    print(f'Wrote {submit_path}')


if __name__ == '__main__':
    main()
