#!/bin/bash
# Set up a virtual environment with all dependencies for AG-RQN.
# Run from the repository root: bash scripts/setup_env.sh
set -e
echo "=================================================================="
echo " AG-RQN environment setup"
echo "=================================================================="

PY=""
for c in python3.11 python3.10 python3.9 python3; do
  if command -v $c >/dev/null 2>&1; then PY=$c; break; fi
done
[ -n "$PY" ] || { echo "FATAL: no python3 found."; exit 1; }
echo "python: $PY  ($($PY --version 2>&1))"

if [ ! -d venv ]; then
  echo "creating venv/ ..."
  $PY -m venv venv
else
  echo "venv/ already exists -- reusing"
fi
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel -q

echo ""
echo "installing torch (cu121) -- the slow part, ~2.5 GB"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "installing the rest..."
pip install -r requirements.txt

echo ""
echo "installing the agrqn package in editable mode..."
pip install -e .

echo ""
echo "=================================================================="
python - <<'PY'
import torch, torchvision, cv2, numpy, pandas, matplotlib, scipy, tqdm, tabulate
print(f"torch        {torch.__version__}")
print(f"torchvision  {torchvision.__version__}")
print(f"opencv       {cv2.__version__}")
print(f"CUDA available : {torch.cuda.is_available()}")
print(f"CUDA version   : {torch.version.cuda}")
print(f"GPUs visible   : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory/1e9:.0f} GB")
PY
echo "=================================================================="

echo ""
echo "pre-downloading DINO ViT-S/16 (so parallel jobs don't race for it)..."
python - <<'PY'
import torch
m = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
print(f"DINO loaded: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params (frozen)")
PY

echo ""
echo "DONE. Every new shell must run:  source venv/bin/activate"
