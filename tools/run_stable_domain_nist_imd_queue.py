import os, signal, subprocess, time
from pathlib import Path
import yaml

ROOT = Path('/group4/zouwenxuan/iml_generalization/repos/ReTraceNet')
PY = Path('/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python')
OUT_ROOT = Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2')
NIST_OUT = OUT_ROOT / 'nist16_stable_bn_v1_full80_20260913'
IMD_OUT = OUT_ROOT / 'imd2020_stable_bn_v1_full50_20260913'
QUEUE_OUT = OUT_ROOT / 'stable_domain_nist_imd_queue_20260913'
QUEUE_OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = QUEUE_OUT / 'summary.yaml'
NIST_SPLIT = OUT_ROOT / 'v3_offline_master_20260912/indomain_NIST16/split_721.json'
IMD_SPLIT = OUT_ROOT / 'v3_offline_master_20260912/indomain_IMD2020/split_721.json'
SCRIPT = ROOT / 'tools/run_v3_stable_bn_indomain.py'
TEMPLATE = OUT_ROOT / 'balanced_residual_v3_full30_20260912/best.pt'
ORIGINAL_MARKER = 'run_nist16_stable_bn_full80.py'

state = {
    'protocol': 'CASIA1_Columbia_tuned_stable_bn',
    'initialization': 'public_content_and_dct_only_task_chain_random',
    'gpu': 0,
    'nist_policy': 'inspect_at_epoch30; stop if no val improvement in epochs21-30, otherwise continue_to50',
    'imd_policy': 'train_to50',
    'status': 'waiting_nist_epoch30',
}

def dump():
    SUMMARY.write_text(yaml.safe_dump(state, sort_keys=False, allow_unicode=True), encoding='utf-8')

def read_progress(out):
    p = out / 'summary.yaml'
    if not p.exists(): return 0, -1, None
    try:
        d = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
        h = d.get('history') or []
        ep = max([int(x.get('epoch', 0)) for x in h] or [0])
        best_ep = int(d.get('best_epoch', -1) or -1)
        return ep, best_ep, d
    except Exception:
        return 0, -1, None

def original_pids():
    out = subprocess.run(['pgrep', '-f', f'[{ORIGINAL_MARKER[0]}]{ORIGINAL_MARKER[1:]}'], text=True, capture_output=True)
    return [int(x) for x in out.stdout.split() if x.isdigit()]

def stop_original():
    for pid in original_pids():
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
    for _ in range(30):
        if not original_pids(): return
        time.sleep(1)
    for pid in original_pids():
        try: os.kill(pid, signal.SIGKILL)
        except ProcessLookupError: pass

def run_dataset(dataset, split, out, epochs, resume=False):
    cmd = [str(PY), str(SCRIPT), '--dataset', dataset, '--split', str(split), '--out', str(out),
           '--template', str(TEMPLATE), '--epochs', str(epochs), '--batch-size', '4', '--seed', '1337']
    if resume: cmd.append('--resume')
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = '0'
    with (QUEUE_OUT / f'{dataset.lower()}_runner.log').open('a', encoding='utf-8') as log:
        return subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode

dump()
while True:
    ep, best_ep, _ = read_progress(NIST_OUT)
    state.update(nist_observed_epoch=ep, nist_best_epoch=best_ep)
    dump()
    if ep >= 30:
        stop_original()
        break
    if not original_pids():
        state['nist_original_process_missing_at_epoch'] = ep
        dump()
        break
    time.sleep(60)

ep, best_ep, _ = read_progress(NIST_OUT)
continue_to50 = ep < 30 or best_ep > 20
target = 50 if continue_to50 else 30
state.update(status='running_nist_final', nist_decision=('continue_to50' if continue_to50 else 'stop_at30_converged'), nist_target_epoch=target)
dump()
rc = run_dataset('NIST16', NIST_SPLIT, NIST_OUT, target, resume=True)
state.update(nist_returncode=rc)
if rc != 0:
    state['status'] = 'failed_nist'; dump(); raise SystemExit(rc)

state['status'] = 'running_imd2020_to50'; dump()
rc = run_dataset('IMD2020', IMD_SPLIT, IMD_OUT, 50, resume=bool((IMD_OUT/'last.pt').exists()))
state.update(imd2020_returncode=rc, status=('completed' if rc == 0 else 'failed_imd2020'))
dump()
raise SystemExit(rc)
