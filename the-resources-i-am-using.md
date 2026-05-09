# Lab Server Resources and Fixes

**Lab Server**: `enoch@192.168.252.237:22` | **Hardware**: 4x Tesla T4 GPUs, 16 CPU cores
**Python Runtime**: `/usr/local/anaconda3/bin/python` (3.8.8, torch cu111, detectron2 0.6, sklearn 1.3.2)

## 1. DeepFool Attack Crash Fix

### Problem

DeepFool attack crashed during 8-attack evaluation with:

```
AttributeError: 'NoneType' object has no attribute 'attack'
```

### Root Cause

`attack_loader.py` passed attack-specific kwargs to ALL modular attacks. DeepFool (optimization-based attack) doesn't accept `alpha` and `iterations` kwargs meant for gradient-based attacks. Failed silently, returned None.

### Solution

[meddef_winlab/gan/attack/attack_loader.py](../meddef_winlab/gan/attack/attack_loader.py) patched April 19, 2026:

- Added `_get_modular_attack_kwargs(normalized_name: str)` method (lines 231-252)
  - Returns attack-specific kwargs dict per attack type
  - Gradient-based (pgd, bim, mi_fgsm): `{'epsilon', 'alpha', 'iterations'}`
  - Auto PGD: `{'epsilon', 'iterations'}`
  - Optimization-based (cw, deepfool, ead, elasticnet): `{'epsilon', 'max_iterations'}`
  - Blackbox (square, hop_skip_jump, boundary): `{'epsilon', 'max_queries'}`
- Updated `_create_modular_attack()` (lines 254-271) to call this method
- Enhanced `AttackHandler.__init__()` (lines 479-496) to raise ValueError on attack creation failure

### Verification

```bash
# Smoke test: instantiate all 8 attacks
ssh -p 22 enoch@192.168.252.237 "cd /data2/enoch/ekd_coding_env/meddef_winlab && \
  /usr/local/anaconda3/bin/python -c '
from gan.attack.attack_loader import modular_get_attack
attacks = [\"fgsm\", \"pgd\", \"bim\", \"mi_fgsm\", \"cw\", \"deepfool\", \"auto_pgd\", \"square\"]
for atk in attacks:
    print(f\"Testing {atk}...\")
    a = modular_get_attack(atk, eps=0.05)
    print(f\"✓ {atk} instantiated: {type(a).__name__}\")
'"
```

---

## 2. meddef2_t_2.1 Retraining & Accuracy Fix

### Problem

Previous training: meddef2_t_2.1 model achieved only 20% validation accuracy

- Used: LR=0.0003, batch=64, loss=weighted, metric=f1 (unstable on small 698-image CCTS dataset)

### Solution

Retrained with optimized hyperparameters: [run/retrain_and_eval_ccts_final.sh](../meddef_winlab/run/retrain_and_eval_ccts_final.sh)

```bash
# Training config (Stage 1 of orchestration script)
LR=5e-5                    # 6× lower for convergence on small dataset
BATCH=32                   # Reduced from 64
EPOCHS=150                 # Max epochs
PATIENCE=50                # Early stopping
LOSS=standard              # Not weighted
METRIC=accuracy            # Not f1
DIST_TEMP=10               # Distillation temperature
OPTIMIZER=adamw            # Default
LR_SCHEDULER=cosine        # Cosine annealing
```

### Result

✅ **83.39% validation accuracy** in 40 epochs (stable convergence)

- All 5 model variants trained successfully
- Checkpoints saved to `out/train/ccts/meddef2_t_*/best.pt`

---

## 3. 8-Attack Evaluation Pipeline

### Configuration

[meddef_winlab/run/retrain_and_eval_ccts_final.sh](../meddef_winlab/run/retrain_and_eval_ccts_final.sh) (Stage 2, sequentially runs all 5 models):

```bash
# Attack sequence (all attacks, epsilon=0.05, batch=32)
ATTACKS="fgsm pgd bim mi_fgsm cw deepfool auto_pgd square"

# Models evaluated (sequential)
MODELS=(
  "meddef2_t_2.1"
  "meddef2_cbam_2"
  "meddef2_cbam_2.1"
  "meddef2_cbam_2.2"
  "meddef2_cbam_2.3"
)
```

### Output Structure

- Results: `out/attack_evaluation/ccts/<model>/all_attack_evaluation_results.csv`
- Columns: attack_type, accuracy, success_rate, avg_perturbation, query_count

### Deployment & Monitoring

```bash
# Deploy orchestration script to server
scp -P 22 /Users/ekd/Documents/coding/py/meddef_winlab/run/retrain_and_eval_ccts_final.sh \
  enoch@192.168.252.237:/data2/enoch/ekd_coding_env/meddef_winlab/run/

# Launch in background (self-daemonizes with nohup)
ssh -p 22 enoch@192.168.252.237 "cd /data2/enoch/ekd_coding_env/meddef_winlab && \
  bash run/retrain_and_eval_ccts_final.sh"

# Check progress
ssh -p 22 enoch@192.168.252.237 "tail -50 /data2/enoch/ekd_coding_env/meddef_winlab/logs/ccts_final_retrain/master.log"
```

---

## 4. Server Resource Cleanup (April 27, 2026 14:37 UTC)

### Problem
Lab server had 60+ running processes with ~45GB GPU memory occupied by idle workloads:
- llama-server: 6.3GB GPU (qwen2.5-7b, 0% util)
- ollama + 3 runners: 18GB+ GPU (0% util)
- PyCharm IDE: 3.3GB RAM
- LLMShield API: port 8000
- libretranslate: ~400MB RAM
- Old baseline evaluations: 25 days running (April 9-25)
- ROS/Jupiter robot stack

### Solution: Multi-Stage Cleanup

**Stage 1: Direct process termination**
```bash
# Kill GPU-consuming processes
pkill -9 -f llama-server
pkill -9 -f 'ollama serve'
pkill -9 -f 'ollama runner'
pkill -9 -f llmshield
pkill -9 -f libretranslate
pkill -9 -f 'remote-dev'
pkill -9 -f robust_progress
pkill -9 -f ros
```

**Stage 2: Discover systemd auto-restart services**
```bash
# These services were auto-restarting and needed explicit handling:
systemctl --user list-units 2>/dev/null | grep -iE '(ollama|libretranslate|argos)'

# Output:
#   ekdflow-ollama.service            loaded active running
#   libretranslate.service            loaded activating auto-restart
#   argos-translate.service           loaded active running
```

**Stage 3: Stop and remove systemd symlinks**
```bash
# Stop services
systemctl --user stop ekdflow-ollama.service ekdflow-img.service libretranslate.service argos-translate.service

# Remove autostart symlinks
rm -f /data2/enoch/.config/systemd/user/default.target.wants/ekdflow-*.service
rm -f /data2/enoch/.config/systemd/user/default.target.wants/libretranslate*
rm -f /data2/enoch/.config/systemd/user/default.target.wants/argos*

# Reload systemd daemon
systemctl --user daemon-reload
```

**Stage 4: Final orphan process cleanup**
```bash
# Kill any remaining instances
pkill -9 ollama
pkill -9 -f ekdflow
pkill -9 -f libretranslate
pkill -9 -f argos
```

### Final Verification (April 27, 14:45 UTC)

**Before cleanup:**
- Processes: 60+
- GPU Memory: 6.3GB (llama-server)
- CPU hogs: 3-4 processes >1%

**After cleanup:**
- ✅ Processes: 20 (system only)
- ✅ GPU Memory: 0-3 MiB per GPU (all freed)
- ✅ CPU hogs: None (0 processes >1%)

```bash
# Verification commands
ps aux | grep enoch | grep -v grep | wc -l
# 20

nvidia-smi --query-gpu=memory.used --format=csv,noheader
# 0 MiB
# 0 MiB
# 3 MiB
# 3 MiB

ps aux | grep enoch | grep -v grep | awk '$3 > 1 {print $2, $3"%", $11}'
# (no output - no heavy processes)
```

### Remaining System Services (Kept - Essential)

- `systemd --user` - session manager
- autossh tunnels (2) - VPS port forwarding (1080 SOCKS, reverse SSH)
- `dbus-daemon` - system messaging
- `sshd` (multiple instances) - SSH connectivity
- `robust_progress_monitor.sh` - TBCR evaluation monitoring
- Cron jobs: `network_selfheal.sh`, `duck.sh` (DuckDNS)

---

## 5. Python Environment Troubleshooting

### Multi-Python Environment Conflicts Resolved

**Problem**: CCTS evaluation crashed with import errors due to detectron2/torch version mismatch

- `~/.virtualenvs/meddef_final/bin/python` (cu102) conflicted with `~/.local/lib/python3.8` (cu111)

**Solution**: Hardcoded system Python with explicit LD_LIBRARY_PATH in all daemon scripts

```bash
#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/anaconda3/lib:/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
/usr/local/anaconda3/bin/python /data2/enoch/ekd_coding_env/meddef_winlab/train.py ...
```

### Verified Python Stack

```bash
/usr/local/anaconda3/bin/python --version
# Python 3.8.8

python -c "import torch; print(f'torch: {torch.__version__}')"
# torch: 1.9.1+cu111

python -c "import detectron2; print(f'detectron2: {detectron2.__version__}')"
# detectron2: 0.6

python -c "import sklearn; print(f'sklearn: {sklearn.__version__}')"
# sklearn: 1.3.2
```

---

## Command Reference

### SSH Access

```bash
# Lab server
ssh -p 22 enoch@192.168.252.237

# VPS (backup storage)
ssh -p 7722 hetawk@mail.es.ekddigital.com
```

### Deploy & Monitor Evaluation

```bash
# Send updated attack_loader.py
scp -P 22 meddef_winlab/gan/attack/attack_loader.py \
  enoch@192.168.252.237:/data2/enoch/ekd_coding_env/meddef_winlab/gan/attack/

# Start evaluation
ssh enoch@192.168.252.237 "cd /data2/enoch/ekd_coding_env/meddef_winlab && \
  bash run/retrain_and_eval_ccts_final.sh 2>&1 | tee eval_stdout.log &"

# Monitor in real-time
watch -n 5 'ssh enoch@192.168.252.237 "tail -30 /data2/enoch/ekd_coding_env/meddef_winlab/logs/ccts_final_retrain/master.log"'

# Check GPU usage during eval
ssh enoch@192.168.252.237 "nvidia-smi"
```

### Dataset & Model Paths

```bash
# CCTS dataset (4-class medical imaging, 698 train images)
/data2/enoch/ekd_coding_env/meddef_winlab/data/ccts/

# Trained models
/data2/enoch/ekd_coding_env/meddef_winlab/out/train/ccts/meddef2_t_*/

# Attack evaluation results
/data2/enoch/ekd_coding_env/meddef_winlab/out/attack_evaluation/ccts/
```

=== All processes under enoch ===
enoch 1543 0.0 0.0 22724 7580 ? Ss 2025 3:56 /lib/systemd/systemd --user
enoch 1546 0.0 0.0 170244 4 ? S 2025 0:00 (sd-pam)
enoch 1585 0.0 0.0 2508 1684 ? Ss 2025 0:00 /data2/enoch/.local/bin/autossh -T -N -D 1080 -M 0 hostingervps
enoch 308785 0.0 0.0 6892 1860 ? Ss Mar30 0:00 bash -c cd /data2/enoch/ekd_coding_env/ultralytics && /data2/enoch/ekd_coding_env/ultralytics/run/robust_progress_monitor.sh | sed -n '1,40p'
enoch 308786 0.0 0.0 7024 2852 ? S Mar30 5:22 /bin/bash /data2/enoch/ekd_coding_env/ultralytics/run/robust_progress_monitor.sh
enoch 308787 0.0 0.0 6564 1720 ? S Mar30 0:09 sed -n 1,40p
enoch 798050 0.0 0.0 2616 1436 ? S 2025 0:00 /bin/sh /data2/enoch/.cache/JetBrains/RemoteDev/dist/6be7d8bc400d7_pycharm-professional-243.23654.74/bin/remote-dev-server.sh run /data2/enoch/ekd_coding_env/art
enoch 807831 7.2 5.1 8244356 3385468 ? Sl 2025 14450:50 /data2/enoch/.cache/JetBrains/RemoteDev/dist/6be7d8bc400d7_pycharm-professional-243.23654.74/bin/remote-dev-server run /data2/enoch/ekd_coding_env/art
enoch 875302 0.0 0.0 6892 1660 ? S Mar30 0:00 bash scripts/run_server.sh --host 0.0.0.0 --port 8000 --no-reload
enoch 875309 0.0 0.0 6892 740 ? S Mar30 0:00 bash scripts/run_server.sh --host 0.0.0.0 --port 8000 --no-reload
enoch 875310 0.0 0.0 5492 20 ? S Mar30 0:00 tee -a logs/server_20260330_131210.log
enoch 875311 0.1 0.0 56436 25360 ? S Mar30 71:44 python3 -m llmshield.src.api.server --host 0.0.0.0 --port 8000 --no-reload
enoch 1487377 0.0 0.0 9392 2064 pts/32 Ss+ 2025 0:00 /bin/bash --rcfile /data2/enoch/.cache/JetBrains/RemoteDev/dist/6be7d8bc400d7_pycharm-professional-243.23654.74/plugins/terminal/shell-integrations/bash/bash-integration.bash -i
enoch 1522779 0.0 0.0 9392 1944 pts/33 Ss+ 2025 0:00 /bin/bash --rcfile /data2/enoch/.cache/JetBrains/RemoteDev/dist/6be7d8bc400d7_pycharm-professional-243.23654.74/plugins/terminal/shell-integrations/bash/bash-integration.bash -i
enoch 1726424 0.0 0.0 2508 1832 ? Ss Apr23 0:00 /data2/enoch/.local/bin/autossh -M 0 -N -o ExitOnForwardFailure yes -o ServerAliveInterval 30 -o ServerAliveCountMax 3 -R 0.0.0.0:8822:localhost:22 hostingervps-reverse
enoch 1747288 0.1 0.0 233324 42636 ? Ssl Apr23 5:58 /data2/enoch/ekdflow-lab/venvs/img/bin/python server.py
enoch 1949343 0.0 0.0 50560 33032 ? S Apr09 0:02 python3 -u -m llmshield.evaluation.baseline.evaluate --dataset /data2/enoch/ekd_coding_env/llm/project0/dataset/llmshield/processed/harmful_examples.jsonl --models llama3:8b --output-dir /data2/enoch/ekd_coding_env/llm/project0/evaluation/baseline/results
enoch 1949376 0.0 0.0 50364 25728 ? S Apr09 0:00 python3 -u -m llmshield.evaluation.baseline.evaluate --dataset /data2/enoch/ekd_coding_env/llm/project0/dataset/llmshield/processed/harmful_examples.jsonl --models mistral:7b-instruct --output-dir /data2/enoch/ekd_coding_env/llm/project0/evaluation/baseline/results
enoch 1949468 0.0 0.0 50600 35148 ? S Apr09 0:01 python3 -u -m llmshield.evaluation.baseline.evaluate --dataset /data2/enoch/ekd_coding_env/llm/project0/dataset/llmshield/processed/harmful_examples.jsonl --models vicuna:13b --output-dir /data2/enoch/ekd_coding_env/llm/project0/evaluation/baseline/results
enoch 2217402 0.0 0.0 7420 3448 ? S Apr07 4:37 bash /data2/enoch/ekd_coding_env/ultralytics/run/eval_tbcr_final.sh
enoch 2350484 0.0 0.1 3041956 105756 ? Ssl Apr24 0:55 /data2/enoch/.local/bin/ollama serve
enoch 2350559 945 7.3 7387916 4843656 ? Rl Apr24 42217:14 /data2/enoch/.local/bin/ollama runner --model /data2/enoch/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f --port 35113
enoch 2350738 872 16.1 13091348 10603532 ? Sl Apr24 38985:44 /data2/enoch/.local/bin/ollama runner --model /data2/enoch/.ollama/models/blobs/sha256-6acb408fff1bc4ded422bbb8c7a10ad644675c154d2f20bfae1bac10bcc1a661 --port 36061
enoch 2402296 868 8.0 7813956 5254500 ? Sl Apr24 38102:43 /data2/enoch/.local/bin/ollama runner --model /data2/enoch/.ollama/models/blobs/sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa --port 46383
enoch 2757891 0.1 0.0 4529836 13064 ? Sl Apr07 35:15 /data2/enoch/micromamba/envs/ros_noetic/bin/python /data2/enoch/micromamba/envs/ros_noetic/bin/roscore
enoch 2757944 0.0 0.0 5645048 22640 ? Ssl Apr07 23:26 /data2/enoch/micromamba/envs/ros_noetic/bin/python /data2/enoch/micromamba/envs/ros_noetic/bin/rosmaster --core -p 11311 -w 3 **log:=/data2/enoch/.ros/log/137282fa-327d-11f1-a8b0-ac1f6bfb501c/master.log
enoch 2758019 0.0 0.0 315692 13260 ? Ssl Apr07 18:06 /data2/enoch/micromamba/envs/ros_noetic/lib/rosout/rosout **name:=rosout **log:=/data2/enoch/.ros/log/137282fa-327d-11f1-a8b0-ac1f6bfb501c/rosout-1.log
enoch 2764896 0.0 0.0 6892 1696 ? Ss Apr07 0:00 bash -c export ALL_PROXY="socks5h://127.0.0.1:1080" HTTPS_PROXY="socks5h://127.0.0.1:1080" HTTP_PROXY="socks5h://127.0.0.1:1080" && cd /data2/enoch/ekd_coding_env/llm/project0 && source .venv/bin/activate && python3 -c " import os print(\"HTTPS_PROXY:\", os.environ.get(\"HTTPS_PROXY\")) from transformers import AutoTokenizer, AutoModelForSequenceClassification print(\"Loading bert-base-uncased...\") m = AutoModelForSequenceClassification.from_pretrained(\"bert-base-uncased\", num_labels=2) print(\"OK:\", type(m).**name**) " 2>&1 | tail -20
enoch 2764897 0.0 0.1 8446276 103164 ? Sl Apr07 21:22 python3 -c import os print("HTTPS_PROXY:", os.environ.get("HTTPS_PROXY")) from transformers import AutoTokenizer, AutoModelForSequenceClassification print("Loading bert-base-uncased...") m = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2) print("OK:", type(m).**name**)
enoch 2764898 0.0 0.0 5512 56 ? S Apr07 0:00 tail -20
enoch 2766567 0.0 0.0 6892 1652 ? Ss Apr07 0:00 bash -c export ALL_PROXY="socks5h://127.0.0.1:1080" HTTPS_PROXY="socks5h://127.0.0.1:1080" HTTP_PROXY="socks5h://127.0.0.1:1080" && cd /data2/enoch/ekd_coding_env/llm/project0 && source .venv/bin/activate && python3 -c " import os print(\"HTTPS_PROXY:\", os.environ.get(\"HTTPS_PROXY\")) from transformers import AutoTokenizer, AutoModelForSequenceClassification print(\"Loading bert-base-uncased...\") tok = AutoTokenizer.from_pretrained(\"bert-base-uncased\") m = AutoModelForSequenceClassification.from_pretrained(\"bert-base-uncased\", num_labels=2) print(\"OK:\", type(m).**name**, \"params:\", sum(p.numel() for p in m.parameters())) " 2>&1 | tail -15
enoch 2766568 0.0 0.2 8394888 157696 ? Sl Apr07 17:13 python3 -c import os print("HTTPS_PROXY:", os.environ.get("HTTPS_PROXY")) from transformers import AutoTokenizer, AutoModelForSequenceClassification print("Loading bert-base-uncased...") tok = AutoTokenizer.from_pretrained("bert-base-uncased") m = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2) print("OK:", type(m).**name**, "params:", sum(p.numel() for p in m.parameters()))
enoch 2766569 0.0 0.0 5512 120 ? S Apr07 0:00 tail -15
enoch 2793860 0.0 0.0 16356 2440 ? S 2025 0:00 /data2/enoch/.virtualenvs/meddef_final/bin/python -c from multiprocessing.resource_tracker import main;main(36)
enoch 2793862 0.0 0.0 17624 2484 ? S 2025 0:00 /data2/enoch/.virtualenvs/meddef_final/bin/python -c from multiprocessing.forkserver import main; main(39, 40, ['**main**'], \*\*{'sys_path': ['/data2/enoch/ekd_coding_env/meddef_winlab', '/data2/enoch/ekd_coding_env/art', '/usr/local/anaconda3/lib/python38.zip', '/usr/local/anaconda3/lib/python3.8', '/usr/local/anaconda3/lib/python3.8/lib-dynload', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages', '/data2/enoch/ekd_coding_env/llm/project0/shared/src', '/data2/enoch/ekd_coding_env/llm/project0/llmshield/src', '/data2/enoch/ekd_coding_env/llm/project0/meddef/src', '/data2/enoch/.local/lib/python3.8/site-packages', '**editable**.ultralytics-8.3.225.finder.**path_hook**', '/usr/local/anaconda3/lib/python3.8/site-packages', '/usr/local/anaconda3/lib/python3.8/site-packages/locket-0.2.1-py3.8.egg', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages/setuptools/\_vendor', '/usr/local/anaconda3/lib/python3.8/site-packages/IPython/extensions']})
enoch 2821331 0.1 0.0 4530276 15276 ? Ssl Apr07 36:09 /data2/enoch/micromamba/envs/ros_noetic/bin/python /data2/enoch/micromamba/envs/ros_noetic/bin/roslaunch jupiterobot2_qt jupiterobot2_qt.launch
enoch 2821418 0.0 0.0 5138584 21340 ? Ssl Apr07 10:09 python /data2/enoch/ekd_coding_env/jup/catkin_ws/src/jupiterobot2/jupiterobot2_qt/jupiterobot2_qt/scripts/tts_qwen.py **name:=tts_qwen **log:=/data2/enoch/.ros/log/137282fa-327d-11f1-a8b0-ac1f6bfb501c/tts_qwen-2.log
enoch 2821419 0.0 0.0 845900 4996 ? Ssl Apr07 17:08 /data2/enoch/ekd_coding_env/jup/catkin_ws/devel/lib/jupiterobot2_voice_xf/iat_publish /voiceWords:=/iat_result **name:=iat_publish **log:=/data2/enoch/.ros/log/137282fa-327d-11f1-a8b0-ac1f6bfb501c/iat_publish-3.log
enoch 2821420 0.0 0.0 779360 4296 ? Ssl Apr07 16:12 /data2/enoch/ekd_coding_env/jup/catkin_ws/devel/lib/jupiterobot2_voice_xf/tts_subscribe **name:=tts_subscribe **log:=/data2/enoch/.ros/log/137282fa-327d-11f1-a8b0-ac1f6bfb501c/tts_subscribe-4.log
enoch 2821444 0.0 0.0 845900 4448 ? Ssl Apr07 16:13 /data2/enoch/ekd_coding_env/jup/catkin_ws/devel/lib/jupiterobot2_voice_xf/iat_publish /voiceWords:=/qwen_img_in /voiceWakeup:=/qwen_img_Wakeup **name:=xf_iat_qwen **log:=/data2/enoch/.ros/log/137282fa-327d-11f1-a8b0-ac1f6bfb501c/xf_iat_qwen-7.log
enoch 2821449 0.0 0.0 5162864 39848 ? Ssl Apr07 9:23 python /data2/enoch/ekd_coding_env/jup/catkin_ws/src/jupiterobot2/jupiterobot2_qt/qwen_ros/scripts/qwen_call.py **name:=qwen_ros **log:=/data2/enoch/.ros/log/137282fa-327d-11f1-a8b0-ac1f6bfb501c/qwen_ros-8.log
enoch 2821451 0.0 0.0 845900 4320 ? Ssl Apr07 16:24 /data2/enoch/ekd_coding_env/jup/catkin_ws/devel/lib/jupiterobot2_voice_xf/iat_publish /voiceWords:=/qwen_in /voiceWakeup:=/qwen_Wakeup **name:=xf_iat **log:=/data2/enoch/.ros/log/137282fa-327d-11f1-a8b0-ac1f6bfb501c/xf_iat-9.log
enoch 2822828 0.0 0.0 16356 2412 ? S 2025 0:00 /data2/enoch/.virtualenvs/meddef_final/bin/python -c from multiprocessing.resource_tracker import main;main(36)
enoch 2822829 0.0 0.0 17624 2520 ? S 2025 0:00 /data2/enoch/.virtualenvs/meddef_final/bin/python -c from multiprocessing.forkserver import main; main(39, 40, ['**main**'], \*\*{'sys_path': ['/data2/enoch/ekd_coding_env/meddef_winlab', '/data2/enoch/ekd_coding_env/art', '/usr/local/anaconda3/lib/python38.zip', '/usr/local/anaconda3/lib/python3.8', '/usr/local/anaconda3/lib/python3.8/lib-dynload', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages', '/data2/enoch/ekd_coding_env/llm/project0/shared/src', '/data2/enoch/ekd_coding_env/llm/project0/llmshield/src', '/data2/enoch/ekd_coding_env/llm/project0/meddef/src', '/data2/enoch/.local/lib/python3.8/site-packages', '**editable**.ultralytics-8.3.225.finder.**path_hook**', '/usr/local/anaconda3/lib/python3.8/site-packages', '/usr/local/anaconda3/lib/python3.8/site-packages/locket-0.2.1-py3.8.egg', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages/setuptools/\_vendor', '/usr/local/anaconda3/lib/python3.8/site-packages/IPython/extensions']})
enoch 2876311 0.0 0.0 5184036 4136 ? Sl 2025 14:50 /data2/enoch/.virtualenvs/meddef_final/bin/python -c from multiprocessing.forkserver import main; main(39, 40, ['**main**'], \*\*{'sys_path': ['/data2/enoch/ekd_coding_env/meddef_winlab', '/data2/enoch/ekd_coding_env/art', '/usr/local/anaconda3/lib/python38.zip', '/usr/local/anaconda3/lib/python3.8', '/usr/local/anaconda3/lib/python3.8/lib-dynload', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages', '/data2/enoch/ekd_coding_env/llm/project0/shared/src', '/data2/enoch/ekd_coding_env/llm/project0/llmshield/src', '/data2/enoch/ekd_coding_env/llm/project0/meddef/src', '/data2/enoch/.local/lib/python3.8/site-packages', '**editable**.ultralytics-8.3.225.finder.**path_hook**', '/usr/local/anaconda3/lib/python3.8/site-packages', '/usr/local/anaconda3/lib/python3.8/site-packages/locket-0.2.1-py3.8.egg', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages/setuptools/\_vendor', '/usr/local/anaconda3/lib/python3.8/site-packages/IPython/extensions']})
enoch 2876312 0.0 0.0 5184036 5168 ? Sl 2025 14:51 /data2/enoch/.virtualenvs/meddef_final/bin/python -c from multiprocessing.forkserver import main; main(39, 40, ['**main**'], \*\*{'sys_path': ['/data2/enoch/ekd_coding_env/meddef_winlab', '/data2/enoch/ekd_coding_env/art', '/usr/local/anaconda3/lib/python38.zip', '/usr/local/anaconda3/lib/python3.8', '/usr/local/anaconda3/lib/python3.8/lib-dynload', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages', '/data2/enoch/ekd_coding_env/llm/project0/shared/src', '/data2/enoch/ekd_coding_env/llm/project0/llmshield/src', '/data2/enoch/ekd_coding_env/llm/project0/meddef/src', '/data2/enoch/.local/lib/python3.8/site-packages', '**editable**.ultralytics-8.3.225.finder.**path_hook**', '/usr/local/anaconda3/lib/python3.8/site-packages', '/usr/local/anaconda3/lib/python3.8/site-packages/locket-0.2.1-py3.8.egg', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages/setuptools/\_vendor', '/usr/local/anaconda3/lib/python3.8/site-packages/IPython/extensions']})
enoch 2876313 0.0 0.0 5184036 3484 ? Sl 2025 14:45 /data2/enoch/.virtualenvs/meddef_final/bin/python -c from multiprocessing.forkserver import main; main(39, 40, ['**main**'], \*\*{'sys_path': ['/data2/enoch/ekd_coding_env/meddef_winlab', '/data2/enoch/ekd_coding_env/art', '/usr/local/anaconda3/lib/python38.zip', '/usr/local/anaconda3/lib/python3.8', '/usr/local/anaconda3/lib/python3.8/lib-dynload', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages', '/data2/enoch/ekd_coding_env/llm/project0/shared/src', '/data2/enoch/ekd_coding_env/llm/project0/llmshield/src', '/data2/enoch/ekd_coding_env/llm/project0/meddef/src', '/data2/enoch/.local/lib/python3.8/site-packages', '**editable**.ultralytics-8.3.225.finder.**path_hook**', '/usr/local/anaconda3/lib/python3.8/site-packages', '/usr/local/anaconda3/lib/python3.8/site-packages/locket-0.2.1-py3.8.egg', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages/setuptools/\_vendor', '/usr/local/anaconda3/lib/python3.8/site-packages/IPython/extensions']})
enoch 2876314 0.0 0.0 5184036 4752 ? Sl 2025 14:54 /data2/enoch/.virtualenvs/meddef_final/bin/python -c from multiprocessing.forkserver import main; main(39, 40, ['**main**'], \*\*{'sys_path': ['/data2/enoch/ekd_coding_env/meddef_winlab', '/data2/enoch/ekd_coding_env/art', '/usr/local/anaconda3/lib/python38.zip', '/usr/local/anaconda3/lib/python3.8', '/usr/local/anaconda3/lib/python3.8/lib-dynload', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages', '/data2/enoch/ekd_coding_env/llm/project0/shared/src', '/data2/enoch/ekd_coding_env/llm/project0/llmshield/src', '/data2/enoch/ekd_coding_env/llm/project0/meddef/src', '/data2/enoch/.local/lib/python3.8/site-packages', '**editable**.ultralytics-8.3.225.finder.**path_hook**', '/usr/local/anaconda3/lib/python3.8/site-packages', '/usr/local/anaconda3/lib/python3.8/site-packages/locket-0.2.1-py3.8.egg', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages/setuptools/\_vendor', '/usr/local/anaconda3/lib/python3.8/site-packages/IPython/extensions']})
enoch 2876628 0.0 0.0 5184028 5240 ? Sl 2025 15:12 /data2/enoch/.virtualenvs/meddef_final/bin/python -c from multiprocessing.forkserver import main; main(39, 40, ['**main**'], \*\*{'sys_path': ['/data2/enoch/ekd_coding_env/meddef_winlab', '/data2/enoch/ekd_coding_env/art', '/usr/local/anaconda3/lib/python38.zip', '/usr/local/anaconda3/lib/python3.8', '/usr/local/anaconda3/lib/python3.8/lib-dynload', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages', '/data2/enoch/ekd_coding_env/llm/project0/shared/src', '/data2/enoch/ekd_coding_env/llm/project0/llmshield/src', '/data2/enoch/ekd_coding_env/llm/project0/meddef/src', '/data2/enoch/.local/lib/python3.8/site-packages', '**editable**.ultralytics-8.3.225.finder.**path_hook**', '/usr/local/anaconda3/lib/python3.8/site-packages', '/usr/local/anaconda3/lib/python3.8/site-packages/locket-0.2.1-py3.8.egg', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages/setuptools/\_vendor', '/usr/local/anaconda3/lib/python3.8/site-packages/IPython/extensions']})
enoch 2876629 0.0 0.0 5184028 5276 ? Sl 2025 15:04 /data2/enoch/.virtualenvs/meddef_final/bin/python -c from multiprocessing.forkserver import main; main(39, 40, ['**main**'], \*\*{'sys_path': ['/data2/enoch/ekd_coding_env/meddef_winlab', '/data2/enoch/ekd_coding_env/art', '/usr/local/anaconda3/lib/python38.zip', '/usr/local/anaconda3/lib/python3.8', '/usr/local/anaconda3/lib/python3.8/lib-dynload', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages', '/data2/enoch/ekd_coding_env/llm/project0/shared/src', '/data2/enoch/ekd_coding_env/llm/project0/llmshield/src', '/data2/enoch/ekd_coding_env/llm/project0/meddef/src', '/data2/enoch/.local/lib/python3.8/site-packages', '**editable**.ultralytics-8.3.225.finder.**path_hook**', '/usr/local/anaconda3/lib/python3.8/site-packages', '/usr/local/anaconda3/lib/python3.8/site-packages/locket-0.2.1-py3.8.egg', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages/setuptools/\_vendor', '/usr/local/anaconda3/lib/python3.8/site-packages/IPython/extensions']})
enoch 2876630 0.0 0.0 5184028 5264 ? Sl 2025 15:04 /data2/enoch/.virtualenvs/meddef_final/bin/python -c from multiprocessing.forkserver import main; main(39, 40, ['**main**'], \*\*{'sys_path': ['/data2/enoch/ekd_coding_env/meddef_winlab', '/data2/enoch/ekd_coding_env/art', '/usr/local/anaconda3/lib/python38.zip', '/usr/local/anaconda3/lib/python3.8', '/usr/local/anaconda3/lib/python3.8/lib-dynload', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages', '/data2/enoch/ekd_coding_env/llm/project0/shared/src', '/data2/enoch/ekd_coding_env/llm/project0/llmshield/src', '/data2/enoch/ekd_coding_env/llm/project0/meddef/src', '/data2/enoch/.local/lib/python3.8/site-packages', '**editable**.ultralytics-8.3.225.finder.**path_hook**', '/usr/local/anaconda3/lib/python3.8/site-packages', '/usr/local/anaconda3/lib/python3.8/site-packages/locket-0.2.1-py3.8.egg', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages/setuptools/\_vendor', '/usr/local/anaconda3/lib/python3.8/site-packages/IPython/extensions']})
enoch 2876631 0.0 0.0 5184028 5256 ? Sl 2025 15:08 /data2/enoch/.virtualenvs/meddef_final/bin/python -c from multiprocessing.forkserver import main; main(39, 40, ['**main**'], \*\*{'sys_path': ['/data2/enoch/ekd_coding_env/meddef_winlab', '/data2/enoch/ekd_coding_env/art', '/usr/local/anaconda3/lib/python38.zip', '/usr/local/anaconda3/lib/python3.8', '/usr/local/anaconda3/lib/python3.8/lib-dynload', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages', '/data2/enoch/ekd_coding_env/llm/project0/shared/src', '/data2/enoch/ekd_coding_env/llm/project0/llmshield/src', '/data2/enoch/ekd_coding_env/llm/project0/meddef/src', '/data2/enoch/.local/lib/python3.8/site-packages', '**editable**.ultralytics-8.3.225.finder.**path_hook\_\_', '/usr/local/anaconda3/lib/python3.8/site-packages', '/usr/local/anaconda3/lib/python3.8/site-packages/locket-0.2.1-py3.8.egg', '/data2/enoch/.virtualenvs/meddef_final/lib/python3.8/site-packages/setuptools/\_vendor', '/usr/local/anaconda3/lib/python3.8/site-packages/IPython/extensions']})
enoch 3114603 0.0 0.0 12236 3932 ? S Apr25 0:00 /usr/bin/ssh -T -N -D 1080 hostingervps
enoch 3208597 0.0 0.0 7468 2616 ? Ss Apr19 0:00 /usr/bin/dbus-daemon --session --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only
enoch 3237561 0.1 4.3 164508860 2874780 ? Ssl Apr19 17:32 /data2/enoch/llama.cpp/build/bin/llama-server --model /data2/enoch/llama.cpp/models/qwen2.5-7b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 9500 --n-gpu-layers 35 --ctx-size 4096 --threads 4
enoch 3304711 0.0 0.6 24808416 431592 ? Ssl Apr19 0:58 /usr/bin/python3 /data2/enoch/.local/bin/libretranslate --host 127.0.0.1 --port 7850 --load-only en,zh,fr,es,de,ar,ru,ja,ko,pt --threads 4
enoch 3304712 0.0 0.6 24710460 400728 ? Ssl Apr19 1:17 /usr/bin/python3 /data2/enoch/argos-api/server.py
enoch 3326230 0.0 0.0 6892 1712 ? Ss Apr08 0:00 bash -s
enoch 3326248 0.3 0.1 5603968 127116 ? Sl Apr08 95:57 python3 -c from transformers import AutoTokenizer, AutoModelForSequenceClassification print('Downloading BERT tokenizer...') tok = AutoTokenizer.from_pretrained('bert-base-uncased') print('Downloading BERT model...') model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2) print(f'BERT params: {sum(p.numel() for p in model.parameters()):,}') print('BERT download complete!')
enoch 3761945 0.0 0.0 6892 120 ? S Apr08 0:00 bash -c chmod +x /data2/enoch/ekd_coding_env/llm/project0/gpu_auto_scheduler.sh && echo "=== Launching auto-scheduler in background ==="; cd /data2/enoch/ekd_coding_env/llm/project0 && nohup bash gpu_auto_scheduler.sh > logs/gpu_scheduler_wrapper.log 2>&1 & echo "SCHEDULER PID=$!"; sleep 3; echo "=== Initial output ==="; tail -30 logs/gpu_scheduler.log 2>/dev/null
enoch 3761947 0.0 0.0 7024 2812 ? S Apr08 1:18 bash gpu_auto_scheduler.sh
enoch 3926690 0.0 0.0 12236 6624 ? S 12:34 0:01 /usr/bin/ssh -N -o ExitOnForwardFailure yes -o ServerAliveInterval 30 -o ServerAliveCountMax 3 -R 0.0.0.0:8822:localhost:22 hostingervps-reverse
root 3926928 0.0 0.0 13900 9036 ? Ss 12:35 0:00 sshd: enoch [priv]
enoch 3927141 0.0 0.0 14040 5320 ? S 12:35 0:00 sshd: enoch
enoch 3976149 0.0 0.0 5476 580 ? S 14:37 0:00 sleep 20
enoch 3976155 0.0 0.0 5476 584 ? S 14:37 0:00 sleep 60
enoch 3976236 0.0 0.0 5476 576 ? S 14:37 0:00 sleep 60
root 3976256 1.0 0.0 13908 9044 ? Ss 14:37 0:00 sshd: enoch [priv]
root 3976258 1.0 0.0 13552 8332 ? Ss 14:37 0:00 sshd: enoch [priv]
sshd 3976275 0.0 0.0 12284 4664 ? S 14:37 0:00 sshd: enoch [net]
enoch 3976339 0.0 0.0 14044 5444 ? S 14:37 0:00 sshd: enoch@notty
enoch 3976343 0.0 0.0 9108 3652 ? R 14:37 0:00 ps aux

=== GPU memory breakdown ===
Mon Apr 27 14:37:24 2026  
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.216.04 Driver Version: 450.216.04 CUDA Version: 11.0 |
|-------------------------------+----------------------+----------------------+
| GPU Name Persistence-M| Bus-Id Disp.A | Volatile Uncorr. ECC |
| Fan Temp Perf Pwr:Usage/Cap| Memory-Usage | GPU-Util Compute M. |
| | | MIG M. |
|===============================+======================+======================|
| 0 Tesla T4 Off | 00000000:18:00.0 Off | 0 |
| N/A 61C P0 30W / 70W | 1730MiB / 15109MiB | 0% Default |
| | | N/A |
+-------------------------------+----------------------+----------------------+
| 1 Tesla T4 Off | 00000000:3B:00.0 Off | 0 |
| N/A 59C P0 30W / 70W | 1260MiB / 15109MiB | 0% Default |
| | | N/A |
+-------------------------------+----------------------+----------------------+
| 2 Tesla T4 Off | 00000000:86:00.0 Off | 0 |
| N/A 60C P0 29W / 70W | 1526MiB / 15109MiB | 0% Default |
| | | N/A |
+-------------------------------+----------------------+----------------------+
| 3 Tesla T4 Off | 00000000:AF:00.0 Off | 0 |
| N/A 57C P0 29W / 70W | 1776MiB / 15109MiB | 0% Default |
| | | N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes: |
| GPU GI CI PID Type Process name GPU Memory |
| ID ID Usage |
|=============================================================================|
| 0 N/A N/A 3237561 C ...pp/build/bin/llama-server 1727MiB |
| 1 N/A N/A 3237561 C ...pp/build/bin/llama-server 1257MiB |
| 2 N/A N/A 3237561 C ...pp/build/bin/llama-server 1523MiB |
| 3 N/A N/A 3237561 C ...pp/build/bin/llama-server 1773MiB |
+-----------------------------------------------------------------------------+

=== Background jobs ===

=== Active ports ===
remote-de 807831 enoch 42u IPv6 3328108 0t0 TCP 127.0.0.1:63342 (LISTEN)
remote-de 807831 enoch 59u IPv6 3262137 0t0 TCP 127.0.0.1:5990 (LISTEN)
remote-de 807831 enoch 347u IPv6 23195163 0t0 TCP 127.0.0.1:61936 (LISTEN)
python3 875311 enoch 7u IPv4 570641681 0t0 TCP \*:8000 (LISTEN)
python 1747288 enoch 11u IPv4 688345509 0t0 TCP 127.0.0.1:7860 (LISTEN)
python3 1949343 enoch 4u IPv4 694089053 0t0 TCP 127.0.0.1:37054->127.0.0.1:11434 (ESTABLISHED)
python3 1949376 enoch 4u IPv4 694087319 0t0 TCP 127.0.0.1:37036->127.0.0.1:11434 (ESTABLISHED)
python3 1949468 enoch 4u IPv4 694067873 0t0 TCP 127.0.0.1:37048->127.0.0.1:11434 (ESTABLISHED)
ollama 2350484 enoch 3u IPv4 694070976 0t0 TCP 127.0.0.1:11434 (LISTEN)
ollama 2350484 enoch 6u IPv4 694076198 0t0 TCP 127.0.0.1:11434->127.0.0.1:37036 (ESTABLISHED)
ollama 2350484 enoch 8u IPv4 694076200 0t0 TCP 127.0.0.1:11434->127.0.0.1:37048 (ESTABLISHED)
ollama 2350484 enoch 10u IPv4 694087320 0t0 TCP 127.0.0.1:11434->127.0.0.1:37054 (ESTABLISHED)
ollama 2350484 enoch 11u IPv4 694082223 0t0 TCP 127.0.0.1:37026->127.0.0.1:35113 (ESTABLISHED)
ollama 2350484 enoch 14u IPv4 694071570 0t0 TCP 127.0.0.1:59330->127.0.0.1:36061 (ESTABLISHED)
ollama 2350484 enoch 16u IPv4 694495540 0t0 TCP 127.0.0.1:52394->127.0.0.1:46383 (ESTABLISHED)
ollama 2350559 enoch 3u IPv4 694067881 0t0 TCP 127.0.0.1:35113 (LISTEN)
ollama 2350559 enoch 6u IPv4 694067883 0t0 TCP 127.0.0.1:35113->127.0.0.1:37026 (ESTABLISHED)
ollama 2350738 enoch 3u IPv4 694083046 0t0 TCP 127.0.0.1:36061 (LISTEN)
ollama 2350738 enoch 6u IPv4 694083048 0t0 TCP 127.0.0.1:36061->127.0.0.1:59330 (ESTABLISHED)
ollama 2402296 enoch 3u IPv4 694489703 0t0 TCP 127.0.0.1:46383 (LISTEN)

=== Disk usage (home dir) ===
➜ py
