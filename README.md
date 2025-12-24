



0. **Set Up Environment and Install Necessary Packages**\
   \
   **Note**: **For memory bandwidth monitoring for merging and splitting tables, use sm110p or other machines with Intel MBM support.**
   - Clone the repository outside dlrm_minrui and follow the instructions provided:
   - For c220g2:
```
     mkdir expr && sudo mkfs.ext4 /dev/sdc && sudo mount /dev/sdc expr && sudo chmod 777 expr
```
   - For c220g5:
```
     mkdir expr && sudo mkfs.ext4 /dev/sdb && sudo mount /dev/sdb expr && sudo chmod 777 expr
```
   - For sm110p:
```
     mkdir expr && sudo mkfs.ext4 /dev/nvme2n1 && sudo mount /dev/nvme2n1 expr && sudo chmod 777 expr
```
  
   - **(Optional)** Install the throttling module (required for memory throttling experiments, optional for embedding table merging/splitting):
```
     git clone https://github.com/RutgersCSSystems/Near-memory.git -b throttle throttle
     cd throttle
     source scripts/setvars.sh
     ./scripts/set_appbench.sh
     cp scripts/gen_config.sh $QUARTZ
```
   - Run the scripts to install packages:
```
     cd expr
     git clone https://github.com/Denker233/dlrm_minrui
     cd dlrm_minrui
     chmod +x *.sh
     ./set_env.sh
     source dlrm_env/bin/activate
     export TMPDIR=$PWD/dlrm_env
     ./install_req.sh
```
   - **Install MLC and intel-cmt-cat**:
     
     From Local (download MLC first):
```
     MLC_TGZ=~/Downloads/mlc_v3.12.tgz
     scp -p "$MLC_TGZ" mt1370@sm110p-10s10616.wisc.cloudlab.us:~/
```
     
     On the remote machine:
```
     ./install_mlc_cmt_cat
```

   - Test run:
```
     python dlrm_s_pytorch.py --mini-batch-size=2 --data-size=6 --debug-mode
```

2. **Prepare and Clean the Dataset**
```
    source dlrm_env/bin/activate
    cd dlrm_minrui/input/
```
   - Download, Untar the dataset file and Rename files:
```
     wget https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz \
       && tar -xzvf criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz \
       && mv train.txt train_original.txt \
       && mv test.txt test_original.txt
```
   - Remove all preprocessed files from the `input` directory to clean up previous runs:
```
     rm -rf kaggleAdDisplayChallenge_processed.npz train_day_* train_fea_*
```

3. **Split Dataset**
   - To split the `train.txt` and `test.txt` files under **dlrm_minrui**:
     - Use the **original dataset**:
```
       cd ..
       python3 input/train_split.py 1
```
     - Use **1/10 of the dataset**:
```
       cd ..
       python3 input/train_split.py 10
```

4. **(Optional) Check File Sizes**
   - To verify the file size manually, use the following command:
```
     du --apparent-size --block-size=1 train.txt | awk '{printf "%.2fG\t%s\n", $1/1073741824, $2}'
     du --apparent-size --block-size=1 test.txt | awk '{printf "%.2fG\t%s\n", $1/1073741824, $2}'
```


6. **Run DLRM with Embedding Table Merging and Splitting**
   
   Example command for testing merge and split strategies:
```bash
   numactl --physcpubind=0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38 --membind=0 \
   python3 dlrm_split_merge_pytorch.py \
     --arch-sparse-feature-size=64 \
     --arch-mlp-bot=13-512-256-64-64 \
     --arch-mlp-top=512-256-1 \
     --data-generation=dataset \
     --data-set=kaggle \
     --raw-data-file=./input/train.txt \
     --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
     --dataset-multiprocessing \
     --loss-function=bce \
     --round-targets=True \
     --mini-batch-size=16380 \
     --print-freq=8192 \
     --print-time \
     --test-mini-batch-size=16380 \
     --test-num-workers=16 \
     --num-indices-per-lookup=64 \
     --merge-emb-threshold=1000 \
     --split-emb-threshold=500000 \
     --num-splits=4 \
     --min-split-rows=100000 \
     --max-split-rows=5000000 \
     --min-table-size=50 \
     --inference-only \
     --nepochs=0
```
   
   Key parameters for merge/split:
   - `--merge-emb-threshold`: Merge tables with size below this threshold (0 = no merging)
   - `--split-emb-threshold`: Split tables with size above this threshold (0 = no splitting)
   - `--num-splits`: Number of splits for large tables (e.g., 2, 4, 6)
   - `--min-table-size`: Minimum table size to consider for merging
   - `--min-split-rows=100000`: Only split tables ≥ 100K rows
   - `--max-split-rows=5000000`: Only split tables ≤ 5M rows

6.1 **Run the automated multi-configuration test script**:
```bash
   ./multi_split_merge.sh
```

## 7. Hot/Cold Runtime Remapping

Split tables based on access patterns - frequently accessed embeddings (hot) vs. rarely accessed (cold).

### Step 1: Profile access patterns
```bash
python dlrm_hot.py \
    --profile-embedding-access \
    --profile-batches=-1 \
    --save-access-profile=./profiles/kaggle_profile_P80_FULL.pkl \
    --hotcold-percentile=80 \
    --hotcold-emb-threshold=1000000 \
    --data-generation=dataset \
    --data-set=kaggle \
    --raw-data-file=./input/train.txt \
    --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
    --loss-function=bce \
    --round-targets=True \
    --mini-batch-size=16384 \
    --arch-sparse-feature-size=64 \
    --arch-mlp-bot=13-512-256-64 \
    --arch-mlp-top=512-256-1 \
    --inference-only \
    --nepochs=0
```

**Key parameters:**
- `--profile-embedding-access`: Enable profiling mode
- `--profile-batches=-1`: Profile full training set 
- `--save-access-profile`: Output file for profile data
- `--hotcold-percentile=80`: Hot embeddings cover 80% of accesses
- `--hotcold-emb-threshold=1000000`: Only split tables ≥ 1M rows

**Output files:**
- `./profiles/kaggle_profile_P80_FULL.pkl` - Raw access counts
- `./profiles/kaggle_profile_P80_FULL_analyzed.pkl` - Analyzed hot/cold mappings

### Step 2: Run inference with runtime remapping
```bash
python dlrm_hot.py \
    --load-access-profile=./profiles/kaggle_profile_P80_FULL.pkl \
    --hotcold-percentile=80 \
    --hotcold-emb-threshold=1000000 \
    --data-generation=dataset \
    --data-set=kaggle \
    --raw-data-file=./input/train.txt \
    --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
    --loss-function=bce \
    --round-targets=True \
    --mini-batch-size=16384 \
    --arch-sparse-feature-size=64 \
    --arch-mlp-bot=13-512-256-64 \
    --arch-mlp-top=512-256-1 \
    --print-freq=100 \
    --print-time \
    --test-mini-batch-size=16384 \
    --test-num-workers=8 \
    --inference-only \
    --nepochs=0
```
## Hot/Cold Preprocessing

Eliminate runtime remapping overhead by preprocessing data once.

### Step 2: Preprocess data (one-time cost)
```bash
# For batch size 16384
python preprocess_hotcold_identical.py \
    --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
    --profile-file=./profiles/kaggle_profile_P80_FULL.pkl \
    --output-dir=./preprocessed_B16384 \
    --batch-size=16384

# For batch size 8192
python preprocess_hotcold_identical.py \
    --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
    --profile-file=./profiles/kaggle_profile_P80_FULL.pkl \
    --output-dir=./preprocessed_B8192 \
    --batch-size=8192
```
### Step 3: Run inference with preprocessed data
```bash
python dlrm_hot.py \
    --use-hotcold-preprocessed \
    --hotcold-preprocessed-dir=./preprocessed_B16384 \
    --mini-batch-size=16384 \
    --arch-sparse-feature-size=64 \
    --arch-mlp-bot=13-512-256-64 \
    --arch-mlp-top=512-256-1 \
    --loss-function=bce \
    --round-targets=True \
    --print-freq=100 \
    --print-time \
    --inference-only \
    --nepochs=0

Run comprehensive comparison of all approaches:
```
chmod +x compare.sh
./compare.sh
```

---

## File Structure
```
dlrm_minrui/
├── dlrm_hot.py                      # Main DLRM implementation
├── preprocess_hotcold_identical.py  # Preprocessing script
├── compare_full.sh                  # Comparison benchmark
├── input/
│   ├── train.txt                    # Raw Criteo data
│   └── kaggleAdDisplayChallenge_processed.npz
├── profiles/
│   ├── kaggle_profile_P80_FULL.pkl           # Raw profile
│   └── kaggle_profile_P80_FULL_analyzed.pkl  # Analyzed profile
├── preprocessed_B16384/             # Preprocessed batch 16384
│   ├── train/
│   │   ├── batch_000000.pt
│   │   └── ...
│   ├── test/
│   └── metadata.pkl
└── comparison_results/              # Benchmark results
    ├── B16384_T80/
    │   ├── baseline.summary.txt
    │   ├── preprocessed.summary.txt
    │   └── runtime.summary.txt
    └── comparison_summary.txt
```
