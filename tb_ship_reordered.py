#!/usr/bin/env python3
"""Convert day_N_reordered.npz to int32 X_cat and ship to the A100 node, incrementally.

Watches for reordered files as preprocessing produces them; for each one:
  1. loads it, converts X_cat float64 -> int32 (value-identical: max id 10M < 2^31,
     verified exact; the training loader collates via np.array(..., dtype=np.int64))
  2. writes day_N_reordered.npz (plain savez) into a staging dir on /mnt/ssd2
  3. rsyncs it to the A100 node
  4. verifies remote size, then deletes the staging copy
After all 24 ship: sends day_fea_count.npz + day_day_count.npz + a .done marker.

Usage: tb_ship_reordered.py <a100_host>            (e.g. cc@192.5.87.x)
Env:   SHIP_COMPRESSED=1  -> savez_compressed instead (0.15 TB total, slower loads)
"""
import os, sys, glob, time, subprocess
import numpy as np

HOST   = sys.argv[1]
SRCDIR = "/home/cc/input/terabyte"
STAGE  = "/mnt/ssd2/ship_stage"
RDEST  = "input/terabyte"          # relative to remote $HOME
COMPRESSED = os.environ.get("SHIP_COMPRESSED", "0") == "1"
os.makedirs(STAGE, exist_ok=True)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def remote_size(rel):
    r = run(f"ssh -o BatchMode=yes {HOST} stat -c %s {RDEST}/{rel} 2>/dev/null")
    return int(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else -1

def ship(local, rel):
    for attempt in (1, 2, 3):
        r = run(f"rsync -a --partial --inplace {local} {HOST}:{RDEST}/{rel}")
        if r.returncode == 0 and remote_size(rel) == os.path.getsize(local):
            return True
        log(f"  rsync attempt {attempt} failed rc={r.returncode}: {r.stderr.strip()[:120]}")
        time.sleep(20)
    return False

# sanity: remote reachable + dest dir
r = run(f"ssh -o BatchMode=yes -o ConnectTimeout=10 {HOST} 'mkdir -p {RDEST} && echo OK'")
if "OK" not in r.stdout:
    print(f"FATAL: cannot reach {HOST}: {r.stderr.strip()}"); sys.exit(1)
log(f"remote {HOST} reachable; compressed={COMPRESSED}")

LEDGER = f"{STAGE}/shipped.txt"
shipped = set()
if os.path.exists(LEDGER):
    shipped = {int(x) for x in open(LEDGER).read().split()}
    log(f"ledger: days already shipped: {sorted(shipped)}")

def mark(i):
    shipped.add(i)
    with open(LEDGER, "a") as f: f.write(f"{i}\n")

pending = None      # (day, staged_path, Popen) of the ship in flight
def wait_pending():
    global pending
    if pending is None: return True
    i, out, proc = pending
    proc.wait()
    ok = proc.returncode == 0 and remote_size(f"day_{i}_reordered.npz") == os.path.getsize(out)
    if not ok:
        # retry synchronously
        log(f"day_{i}: async ship failed, retrying synchronously")
        ok = ship(out, f"day_{i}_reordered.npz")
    if ok:
        os.remove(out); mark(i); log(f"day_{i}: shipped + verified  ({len(shipped)}/24)")
    else:
        log(f"day_{i}: SHIP FAILED after retries"); sys.exit(1)
    pending = None
    return ok

while len(shipped) < 24:
    for i in range(24):
        if i in shipped: continue
        src = f"{SRCDIR}/day_{i}_reordered.npz"
        if not os.path.exists(src): continue          # dangling symlink until written
        # make sure the writer is done with it: size stable across 60 s
        s1 = os.path.getsize(src); time.sleep(60); s2 = os.path.getsize(src)
        if s1 != s2 or s1 < 1e9:
            log(f"day_{i}: still being written ({s1} -> {s2}), waiting"); continue
        log(f"day_{i}: converting to int32 ({s1/2**30:.1f} GiB in)")
        t0 = time.time()
        with np.load(src) as d:
            X_cat = d["X_cat"]; X_int = d["X_int"]; y = d["y"]
        # all three arrays arrive float64 (they pass through float64 intermediates).
        # Convert to int32 with a safety check: values must be integral and in range.
        def to_i32(a, name):
            i = a.astype(np.int32)
            # full round-trip equality: every value must survive int32 exactly
            if not np.array_equal(i.astype(a.dtype), a):
                log(f"  {name}: NOT int32-safe, keeping original dtype")
                return a
            return i
        X_cat = to_i32(X_cat, "X_cat"); X_int = to_i32(X_int, "X_int"); y = to_i32(y, "y")
        out = f"{STAGE}/day_{i}_reordered.npz"
        (np.savez_compressed if COMPRESSED else np.savez)(out, X_cat=X_cat, X_int=X_int, y=y)
        del X_cat, X_int, y
        log(f"day_{i}: wrote {os.path.getsize(out)/2**30:.1f} GiB in {time.time()-t0:.0f}s; shipping async")
        wait_pending()                      # previous ship must finish first
        proc = subprocess.Popen(
            f"rsync -a --partial --inplace {out} {HOST}:{RDEST}/day_{i}_reordered.npz",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pending = (i, out, proc)
    if len(shipped) < 24:
        time.sleep(120)

wait_pending()
for f in ("day_fea_count.npz", "day_day_count.npz"):
    p = f"{SRCDIR}/{f}"
    if not ship(p, f):
        log(f"FATAL: could not ship {f}"); sys.exit(1)
    log(f"shipped {f}")
run(f"ssh {HOST} 'touch {RDEST}/day_transfer.done'")
log("ALL 24 DAYS + METADATA SHIPPED — day_transfer.done set")
