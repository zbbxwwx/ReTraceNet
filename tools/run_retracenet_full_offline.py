"""Pure local ReTraceNet state machine. No network, API, LLM, or agent calls."""
import argparse, datetime, fcntl, json, os, shlex, signal, subprocess, sys, time, traceback
from pathlib import Path
import yaml

BASE=Path('/group4/zouwenxuan/iml_generalization')
REPO=BASE/'repos/ReTraceNet'; PY=BASE/'envs/mesorch/bin/python'
ROOT=BASE/'outputs/retracenet_v2/offline_master_20260912'
FINAL=BASE/'outputs/retracenet_v2/final_experiment_summary_20260912'
E24=BASE/'outputs/retracenet_v2/external_four_e24_same_protocol_20260912'
RECAL=FINAL/'01_e24_recalibration'; INV=FINAL/'data_inventory'; FAIR=BASE/'outputs/retracenet_v2/fair_training_v1_full_20260911'
TOOLS=BASE/'tools'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,data,yaml_mode=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        (yaml.safe_dump(data,f,sort_keys=False) if yaml_mode else json.dump(data,f,indent=2));f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def load(path,default=None):
    try:
        p=Path(path);return yaml.safe_load(p.read_text()) if p.suffix in ['.yaml','.yml'] else json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError,yaml.YAMLError):return default
def log(msg,error=False):
    line=f'{now()} {msg}\n';p=ROOT/('errors.log' if error else 'master.log')
    with p.open('a') as f:f.write(line);f.flush();os.fsync(f.fileno())
    print(line,end='',flush=True)
def alive(pid):
    try:os.kill(int(pid),0);return True
    except (OSError,TypeError,ValueError):return False
def status_file(path):return load(path,{}) or {}
def command(stage,args,outdir,marker='state.json',env=None):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True);mark=outdir/marker
    m=status_file(mark)
    if m.get('status')=='completed':return 'completed'
    state=load(ROOT/'state.json',{})
    child=state.get('child',{})
    if child.get('stage')==stage and alive(child.get('pid')):return 'running'
    if child.get('stage')==stage and not alive(child.get('pid')):
        m=status_file(mark)
        if m.get('status')=='completed':return 'completed'
        raise RuntimeError(f'{stage} child exited without completed marker: {m}')
    cmd=[str(PY),'-u',*map(str,args)]
    with (ROOT/'commands.log').open('a') as f:f.write(now()+' '+shlex.join(cmd)+'\n');f.flush();os.fsync(f.fileno())
    stdout=(outdir/'stdout.log').open('a');stderr=(outdir/'stderr.log').open('a')
    runenv=os.environ.copy();runenv.update({'CUDA_VISIBLE_DEVICES':'0','PYTHONUNBUFFERED':'1'});runenv.update(env or {})
    proc=subprocess.Popen(cmd,cwd=REPO,stdout=stdout,stderr=stderr,env=runenv,start_new_session=True)
    state.update(stage=stage,status='running',updated_at=now(),child={'stage':stage,'pid':proc.pid,'output_dir':str(outdir),'command':cmd})
    atomic(ROOT/'state.json',state);log(f'started {stage} pid={proc.pid}')
    return 'running'
def transition(stage,reason,**extra):
    state=load(ROOT/'state.json',{});state.update(stage=stage,status='waiting',updated_at=now(),reason=reason,child={});state.update(extra)
    atomic(ROOT/'state.json',state);log(f'transition {stage}: {reason}')
def decision(stage,route,evidence,reason,rejected):
    row={'time':now(),'stage':stage,'route':route,'evidence':evidence,'reason':reason,'rejected_routes':rejected}
    with (ROOT/'decisions.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    return row
def best(path):
    d=load(Path(path)/'state.json') or load(Path(path)/'summary.yaml');return float(d['best_F1']),d['best_checkpoint']

def proxy_accept(control,treatment,route,parent_off=.14896679514144434):
    c=load(Path(control)/'summary.json');t=load(Path(treatment)/'summary.json')
    deltas={k:t['conditions'][k]['fake_locked_F1']-c['conditions'][k]['fake_locked_F1'] for k in t['conditions'] if k not in ['clean','dct_off']}
    multiple=sum(v>0 for v in deltas.values())>=3
    ok=t['clean_fake_F1_best']>=.830 and t['robustness_macro_F1']>=c['robustness_macro_F1']+.010 and multiple
    if route=='G-A':ok=ok and t['dct_off_fake_F1_best']>=parent_off+.05
    return ok,{'control_macro':c['robustness_macro_F1'],'treatment_macro':t['robustness_macro_F1'],
               'clean':t['clean_fake_F1_best'],'dct_off':t['dct_off_fake_F1_best'],'condition_deltas':deltas,'multiple':multiple}

def tick():
    s=load(ROOT/'state.json',{'stage':'WAIT_E24_EXTERNAL'})
    st=s['stage']
    if st=='WAIT_E24_EXTERNAL':
        q=load(E24/'summary.yaml',{})
        if q.get('status')=='completed':transition('E24_THRESHOLD_RECALIBRATION','existing E24 external complete')
        elif q.get('status')=='failed':raise RuntimeError('E24 external failed: '+str(q))
        elif alive(239653):log('waiting existing E24 external pid239653')
        else:raise RuntimeError('E24 external pid gone without complete/failed marker')
    elif st=='E24_THRESHOLD_RECALIBRATION':
        q=load(RECAL/'state.json',{})
        if q.get('status')=='completed':transition('GENERALIZATION_DIAGNOSIS','recalibration and external finalize complete')
        elif q.get('status')=='failed':raise RuntimeError(str(q))
        elif alive(240115):log('waiting existing recalibration pid240115')
        else:
            r=command(st,[REPO/'tools/takeover_recalibrate.py'],RECAL)
    elif st=='GENERALIZATION_DIAGNOSIS':
        d=load(FINAL/'generalization_route_decision.json')
        if not d:raise RuntimeError('missing deterministic route decision')
        route=d['route']
        if route=='G-C':
            # Deterministic fallback: multiple low-oracle domains imply broad shift;
            # a codec-only claim requires >=3 aligned codec-associated advantages.
            if len(d.get('both_oracle_below_0p5',[]))>=2:route='G-B'
            else:
                decision(st,'STOP_NEEDS_REVIEW',d,'G-C not reliably reducible by frozen rules',['guess codec route'])
                transition('STOP_NEEDS_REVIEW','ambiguous mixed external shift');return
        target='PILOT_A_CONTROL' if route=='G-A' else 'PILOT_B_CONTROL'
        decision(st,route,d,d['reason'],d.get('rejected_routes',[]));transition(target,'deterministic generalization route',route=route)
    elif st in ['PILOT_A_CONTROL','PILOT_B_CONTROL']:
        out=ROOT/'pilot_control';r=command(st,[REPO/'tools/takeover_train.py','--out',out,'--strategy','control'],out)
        if r=='completed':transition('PILOT_A_TREATMENT' if st.startswith('PILOT_A') else 'PILOT_B_TREATMENT','matched control completed',route=s['route'])
    elif st in ['PILOT_A_TREATMENT','PILOT_B_TREATMENT']:
        route=s['route'];strategy='dct_dropout' if route=='G-A' else 'paired';out=ROOT/('pilot_dct_dropout' if route=='G-A' else 'pilot_paired_lam0p1')
        args=[REPO/'tools/takeover_train.py','--out',out,'--strategy',strategy]
        if strategy=='paired':args += ['--lambda-cons','0.1']
        r=command(st,args,out)
        if r=='completed':transition('PILOT_PROXY_CONTROL','treatment completed',route=route,treatment_dir=str(out))
    elif st=='PILOT_PROXY_CONTROL':
        _,ck=best(ROOT/'pilot_control');out=ROOT/'proxy_control';r=command(st,[REPO/'tools/takeover_proxy.py','--checkpoint',ck,'--out',out],out,marker='summary.json')
        if r=='completed':transition('PILOT_PROXY_TREATMENT','control proxy complete',route=s['route'],treatment_dir=s['treatment_dir'])
    elif st=='PILOT_PROXY_TREATMENT':
        _,ck=best(s['treatment_dir']);out=Path(s['treatment_dir'])/'proxy';r=command(st,[REPO/'tools/takeover_proxy.py','--checkpoint',ck,'--out',out],out,marker='summary.json')
        if r=='completed':transition('PILOT_EVALUATION','treatment proxy complete',route=s['route'],treatment_dir=s['treatment_dir'])
    elif st=='PILOT_EVALUATION':
        ok,ev=proxy_accept(ROOT/'proxy_control',Path(s['treatment_dir'])/'proxy',s['route'])
        decision(st,'PASS' if ok else 'FAIL',ev,'frozen pilot gates evaluated',['external test selection'])
        if ok:transition('GENERALIZED_FULL_TRAIN','pilot passed',strategy='dct_dropout' if s['route']=='G-A' else 'paired',lambda_cons=.1)
        elif s['route']=='G-B' and not s.get('retry_done'):
            t=load(Path(s['treatment_dir'])/'proxy/summary.json');lam=.05 if t['clean_fake_F1_best']<.830 else .2
            transition('PILOT_B_RETRY_TRAIN','single allowed lambda retry',route='G-B',lambda_cons=lam,retry_done=True)
        elif s['route']=='G-A':transition('PILOT_B_TREATMENT','A failed; broad external weakness permits one B attempt',route='G-B',treatment_dir=str(ROOT/'pilot_paired_lam0p1'))
        else:transition('INDOMAIN_DATA_AUDIT','generalization pilots stopped after frozen failures',generalized_checkpoint=None)
    elif st=='PILOT_B_RETRY_TRAIN':
        out=ROOT/f"pilot_paired_lam{str(s['lambda_cons']).replace('.','p')}";r=command(st,[REPO/'tools/takeover_train.py','--out',out,'--strategy','paired','--lambda-cons',s['lambda_cons']],out)
        if r=='completed':transition('PILOT_B_RETRY_PROXY','retry train complete',route='G-B',lambda_cons=s['lambda_cons'],treatment_dir=str(out),retry_done=True)
    elif st=='PILOT_B_RETRY_PROXY':
        _,ck=best(s['treatment_dir']);out=Path(s['treatment_dir'])/'proxy';r=command(st,[REPO/'tools/takeover_proxy.py','--checkpoint',ck,'--out',out],out,marker='summary.json')
        if r=='completed':
            ok,ev=proxy_accept(ROOT/'proxy_control',out,'G-B');decision(st,'PASS' if ok else 'FAIL',ev,'single retry frozen gates',['further lambda search'])
            transition('GENERALIZED_FULL_TRAIN' if ok else 'INDOMAIN_DATA_AUDIT','retry evaluated',strategy='paired' if ok else None,lambda_cons=s['lambda_cons'],generalized_checkpoint=None)
    elif st=='GENERALIZED_FULL_TRAIN':
        out=ROOT/'generalized_full';args=[REPO/'tools/takeover_train.py','--out',out,'--strategy',s['strategy'],'--public-init','--epochs','30','--lr','.001','--seed','3407']
        if s['strategy']=='paired':args += ['--lambda-cons',s['lambda_cons']]
        r=command(st,args,out)
        if r=='completed':transition('GENERALIZED_PROXY','formal training complete',strategy=s['strategy'])
    elif st=='GENERALIZED_PROXY':
        _,ck=best(ROOT/'generalized_full');out=ROOT/'generalized_proxy';r=command(st,[REPO/'tools/takeover_proxy.py','--checkpoint',ck,'--out',out],out,marker='summary.json')
        if r=='completed':transition('GENERALIZED_EXTERNAL_EVAL','formal proxy complete',generalized_checkpoint=ck)
    elif st=='GENERALIZED_EXTERNAL_EVAL':
        out=FINAL/'generalized_external_four';r=command(st,[REPO/'tools/takeover_external_checkpoint.py','--checkpoint',s['generalized_checkpoint'],'--out',out],out,marker='decision.json')
        if r=='completed':transition('INDOMAIN_DATA_AUDIT','generalization track complete',generalized_checkpoint=s['generalized_checkpoint'])
    elif st=='INDOMAIN_DATA_AUDIT':
        q=load(INV/'state.json',{})
        if q.get('status')=='completed':transition('INDOMAIN_SPLITS','inventory complete')
        elif q.get('status')=='failed':raise RuntimeError(str(q))
        elif alive(240352):log('waiting existing data inventory pid240352')
        else:r=command(st,[REPO/'tools/takeover_data_inventory.py'],INV)
    elif st=='INDOMAIN_SPLITS':
        # Exact duplicate information is ready, but scripts deliberately do not
        # promote filename guesses to source truth. Stop only under user's named
        # ambiguity condition rather than manufacture a leaking split.
        inv=load(INV/'inventory_summary.json')
        unverified=[k for k,v in inv.items() if not v.get('source_isolation_verified')]
        if unverified:
            decision(st,'STOP_NEEDS_REVIEW',{'unverified_source_relations':unverified},
                     'Cannot certify source-disjoint group split from files alone',['random filename split'])
            transition('STOP_NEEDS_REVIEW','source/group relations require authoritative mapping',unverified=unverified)
        else:raise RuntimeError('split implementation not reached: verified manifests unexpectedly present')
    elif st=='STOP_NEEDS_REVIEW':
        atomic(ROOT/'final_status.yaml',{'status':'STOP_NEEDS_REVIEW','stage':st,'reason':s.get('reason'),'updated_at':now()},True)
        return 'stop'
    elif st=='DONE':return 'done'
    else:raise RuntimeError('unknown stage '+st)

def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--poll-seconds',type=int,default=300);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);lock=(ROOT/'lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('offline supervisor already running')
    (ROOT/'pid').write_text(str(os.getpid()));
    if not (ROOT/'state.json').exists():atomic(ROOT/'state.json',{'stage':'WAIT_E24_EXTERNAL','status':'waiting','created_at':now(),'child':{}})
    if a.dry_run:
        print(json.dumps({'state':load(ROOT/'state.json'),'python':str(PY),'repo':str(REPO),'network_or_llm':False,'poll_seconds':a.poll_seconds},indent=2));return
    log('offline supervisor started; no network/API/LLM dependencies')
    while True:
        try:
            result=tick()
            if result in ['done','stop']:break
        except BaseException:
            err=traceback.format_exc();log(err,error=True);s=load(ROOT/'state.json',{});s.update(status='STOP_NEEDS_REVIEW',error=err,updated_at=now());atomic(ROOT/'state.json',s);atomic(ROOT/'final_status.yaml',s,True);break
        time.sleep(max(5,a.poll_seconds))

if __name__=='__main__':main()
