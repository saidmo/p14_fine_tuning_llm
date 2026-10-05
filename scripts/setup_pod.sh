#!/usr/bin/env bash
# Préparation du pod RunPod pour l'entraînement (SFT / DPO).
#
# Idempotent : à relancer après CHAQUE redémarrage du pod. Le disque conteneur est
# effacé à l'arrêt ; seul le volume /workspace est conservé. Ce script :
#   1. réinstalle les outils système manquants (tmux) ;
#   2. crée, si besoin, un environnement virtuel sur le volume, qui RÉUTILISE le
#      PyTorch de l'image (--system-site-packages) au lieu de le retélécharger ;
#   3. installe les dépendances d'entraînement sans jamais remplacer ce PyTorch ;
#   4. vérifie GPU, bf16, bibliothèques et jeton Hugging Face.
#
# Usage (depuis la racine du dépôt cloné dans /workspace) :
#   bash scripts/setup_pod.sh
#   source /workspace/venv/bin/activate
set -euo pipefail

WORKSPACE=/workspace
VENV="$WORKSPACE/venv"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "== 1. Outils système"
if ! command -v tmux >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq tmux >/dev/null
fi
echo "tmux : $(tmux -V)"

echo "== 2. Environnement virtuel ($VENV)"
if [ ! -x "$VENV/bin/python" ]; then
    python -m venv --system-site-packages "$VENV"
    echo "créé (réutilise le PyTorch de l'image)"
else
    echo "déjà présent"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "== 3. Dépendances d'entraînement"
TORCH_VERSION="$(python -c 'import torch; print(torch.__version__)')"
CONSTRAINTS="$(mktemp)"
echo "torch==${TORCH_VERSION}" > "$CONSTRAINTS"      # pip ne peut pas remplacer le PyTorch de l'image
python -m pip install -q --upgrade pip
python -m pip install -q -r "$REPO_DIR/requirements-train.txt" -c "$CONSTRAINTS"
rm -f "$CONSTRAINTS"

echo "== 4. Vérifications"
mkdir -p "${HF_HOME:-$WORKSPACE/hf_cache}" "$WORKSPACE/mlflow" "$WORKSPACE/outputs"
python - <<'EOF'
import os
import torch, transformers, trl, peft, accelerate, datasets, mlflow
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) | transformers {transformers.__version__} | "
      f"trl {trl.__version__} | peft {peft.__version__} | accelerate {accelerate.__version__} | "
      f"datasets {datasets.__version__} | mlflow {mlflow.__version__}")
assert torch.cuda.is_available(), "GPU non disponible"
free, total = torch.cuda.mem_get_info()
print(f"GPU : {torch.cuda.get_device_name(0)} — {free / 2**30:.0f} / {total / 2**30:.0f} Go libres — "
      f"bf16 : {torch.cuda.is_bf16_supported()}")
print(f"HF_HOME : {os.environ.get('HF_HOME', '(non défini)')}")
print(f"HF_TOKEN : {'présent' if os.environ.get('HF_TOKEN') else 'ABSENT (hf auth login)'}")
EOF
echo "Prêt. Activer l'environnement : source $VENV/bin/activate"
