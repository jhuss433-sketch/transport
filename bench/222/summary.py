import glob, re, subprocess, statistics as st
def t_ms(log):
    out = subprocess.run(['snpe-diagview','--input_log',log], capture_output=True, text=True).stdout
    m = re.search(r'Total Inference Time:\s+(\d+) us', out); return int(m.group(1))/1000 if m else None
def pss_mib(f):
    tot = [sum(int(x) for x in re.findall(r'^Pss:\s+(\d+) kB', s, re.M)) for s in open(f).read().split('====')]
    tot = [t for t in tot if t > 0]; return sum(tot)/len(tot)/1024 if tot else None
print(f"{'Runtime':<20}{'runs (ms)':<28}{'median ms':>10}{'avg PSS MiB':>13}")
for res in ['jacob_222', 'jacob_222_gpu']:
    run = sorted(glob.glob(f'{res}/2026-*'))[-1]
    for rt in sorted(glob.glob(f'{run}/timing/*')):
        name = rt.split('/')[-1]
        ts = [t for t in (t_ms(l) for l in sorted(glob.glob(f'{rt}/Run*/SNPEDiag_0.log'))) if t]
        ms = [m for m in (pss_mib(f) for f in glob.glob(f'{run}/mem/{name}/Run*/MemLog.txt')) if m]
        print(f"{name:<20}{str([round(t,2) for t in ts]):<28}{st.median(ts):>10.2f}{(sum(ms)/len(ms) if ms else float('nan')):>13.2f}")
