#!/usr/bin/env python3
"""Train DLRM on Avazu_x1 (standalone loop; reuses DLRM_Net from dlrm_s_pytorch).

No dense features -> a single constant zero dense input through a minimal bottom MLP
(contributes one learned vector; the model's signal is the 22 embeddings).
Config mirrors CAFE's avazu.sh: D=16, top 512-256-1, SGD lr 0.1, batch 128, 1 epoch.
Saves models/dlrm_avazu_x1.pt (state_dict + config).
"""
import os, sys, time, numpy as np, torch
sys.path.insert(0, '/home/cc/expr/dlrm_minrui'); os.chdir('/home/cc/expr/dlrm_minrui')
from sklearn.metrics import roc_auc_score
from dlrm_s_pytorch import DLRM_Net

D, BATCH, LR = 16, 128, 0.1
EPOCHS = 1
torch.set_num_threads(24)
t0=time.time()
def log(m): print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)

z = np.load('/mnt/nvme1/avazu/avazu_x1.npz')
Xtr, ytr = z['X_cat_train'], z['y_train']
Xte, yte = z['X_cat_test'],  z['y_test']
counts = z['counts']
ntab = len(counts)
log(f"train {len(ytr):,} test {len(yte):,} tables {ntab} total-rows {counts.sum():,}")

ln_emb = np.array(counts)
ln_bot = np.array([1, 64, D])
num_int = (ntab+1)*ntab//2 + D
ln_top = np.array([num_int, 512, 256, 1])
dlrm = DLRM_Net(D, ln_emb, ln_bot, ln_top, arch_interaction_op="dot",
                arch_interaction_itself=False, sigmoid_bot=-1,
                sigmoid_top=ln_top.size-2, loss_function="bce")
opt = torch.optim.SGD(dlrm.parameters(), lr=LR)
loss_fn = torch.nn.BCELoss()
log(f"model built: emb {sum(int(c)*D*4 for c in counts)/2**20:.1f} MB fp32")

n = len(ytr); nbatch = n // BATCH
dense = torch.zeros(BATCH, 1)
off = torch.arange(BATCH)
perm = np.random.default_rng(0).permutation(n)   # shuffle once
for ep in range(EPOCHS):
    running = 0.0
    for j in range(nbatch):
        idx = perm[j*BATCH:(j+1)*BATCH]
        xc = torch.from_numpy(Xtr[idx].astype(np.int64))
        yb = torch.from_numpy(ytr[idx].astype(np.float32)).view(-1,1)
        lS_i = [xc[:,t] for t in range(ntab)]
        lS_o = [off]*ntab
        Z = dlrm(dense, lS_o, lS_i)
        loss = loss_fn(Z, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item()
        if (j+1) % 20000 == 0:
            log(f"it {j+1}/{nbatch}  loss {running/20000:.5f}")
            running = 0.0

torch.save({'state_dict': dlrm.state_dict(), 'counts': counts, 'D': D},
           'models/dlrm_avazu_x1.pt')
log("weights saved (pre-eval) -> models/dlrm_avazu_x1.pt")
log("evaluating on test (8.1M rows) ...")
dlrm.eval(); scores=[]
TB = 16384
dte = torch.zeros(TB,1); offe = torch.arange(TB)
with torch.no_grad():
    for j in range(0, len(yte)//TB*TB, TB):
        xc = torch.from_numpy(Xte[j:j+TB].astype(np.int64))
        Z = dlrm(dte, [offe]*ntab, [xc[:,t] for t in range(ntab)])
        scores.append(Z.numpy().ravel())
ncov = len(scores)*TB
auc = roc_auc_score(yte[:ncov], np.concatenate(scores))
log(f"TEST AUC = {auc:.6f}   (FuxiCTR reference DLRM-family models: ~0.76-0.77)")
torch.save({'state_dict': dlrm.state_dict(), 'counts': counts, 'D': D,
            'test_auc': float(auc)}, 'models/dlrm_avazu_x1.pt')
log("saved models/dlrm_avazu_x1.pt (with test_auc)")
