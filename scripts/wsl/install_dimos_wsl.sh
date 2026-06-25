#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$HOME/dimos}"
SOURCE_REPO="${2:-}"
EXTRAS="${DIMOS_WSL_EXTRAS:-base,cuda,sim,misc}"
PYTHON_BIN="${DIMOS_WSL_PYTHON:-3.12}"

verify_python_module() {
    local module_name="$1"
    "$REPO_DIR/.venv/bin/python" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$module_name') is not None else 1)"
}

sync_from_source() {
    local source_repo="$1"
    local repo_dir="$2"

    echo "[dimos-wsl] syncing local checkout from $source_repo into $repo_dir"
    mkdir -p "$repo_dir"
    rsync -a --delete \
        --exclude '.git/' \
        --exclude '.venv/' \
        --exclude '.dimos_runtime/' \
        --exclude 'logs/' \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        --exclude 'node_modules/' \
        --exclude 'data/.lfs/' \
        --exclude 'data/**/.lfs/' \
        "$source_repo/" "$repo_dir/"
}

export DEBIAN_FRONTEND=noninteractive

sudo apt-get update
sudo apt-get install -y \
    build-essential \
    curl \
    git \
    git-lfs \
    libegl1 \
    libgl1 \
    libglib2.0-dev \
    libturbojpeg \
    libturbojpeg0-dev \
    pkg-config \
    portaudio19-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    rsync

if [[ -n "$SOURCE_REPO" ]] && [[ -d "$SOURCE_REPO" ]]; then
    if [[ ! -d "$REPO_DIR/.git" ]]; then
        echo "[dimos-wsl] cloning repo metadata into $REPO_DIR before syncing local checkout"
        git clone https://github.com/dimensionalOS/dimos.git "$REPO_DIR"
    fi
    sync_from_source "$SOURCE_REPO" "$REPO_DIR"
else
    if [[ -d "$REPO_DIR/.git" ]]; then
        echo "[dimos-wsl] updating existing repo at $REPO_DIR"
        git -C "$REPO_DIR" pull --ff-only
    else
        echo "[dimos-wsl] cloning repo into $REPO_DIR"
        git clone https://github.com/dimensionalOS/dimos.git "$REPO_DIR"
    fi
fi

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$PATH"

cd "$REPO_DIR"

git lfs install --local

mobileclip_archive="$REPO_DIR/data/.lfs/models_mobileclip.tar.gz"
if [[ -f "$mobileclip_archive" && ! -s "$mobileclip_archive" ]]; then
    echo "[dimos-wsl] removing empty MobileCLIP archive before refetch"
    rm -f "$mobileclip_archive"
fi

git lfs pull --include 'data/.lfs/models_mobileclip.tar.gz'

if [[ ! -s "$mobileclip_archive" ]]; then
    echo "[dimos-wsl] ERROR: MobileCLIP archive is missing or empty after git lfs pull" >&2
    exit 1
fi

MOBILECLIP_ARCHIVE="$mobileclip_archive" python3 - <<'PY'
import os
import tarfile
from pathlib import Path
archive = Path(os.environ["MOBILECLIP_ARCHIVE"])
with tarfile.open(archive, "r:gz") as tar:
    first = tar.next()
    if first is None:
        raise SystemExit("[dimos-wsl] ERROR: MobileCLIP archive contains no members")
PY

if [[ ! -d .venv ]]; then
    uv venv --python "$PYTHON_BIN"
fi

read -r -a extra_array <<< "${EXTRAS//,/ }"
uv_sync_args=(--no-default-groups)
for extra_name in "${extra_array[@]}"; do
    [[ -n "$extra_name" ]] || continue
    uv_sync_args+=(--extra "$extra_name")
done

uv sync "${uv_sync_args[@]}"

if ! verify_python_module open_clip; then
    echo "[dimos-wsl] open_clip missing after uv sync; installing open_clip_torch directly"
    uv pip install open_clip_torch==3.2.0
fi

verify_python_module open_clip || {
    echo "[dimos-wsl] ERROR: open_clip is still unavailable in $REPO_DIR/.venv after setup" >&2
    exit 1
}

echo
echo "[dimos-wsl] install complete"
echo "[dimos-wsl] repo: $REPO_DIR"
echo "[dimos-wsl] run example:"
echo "  cd $REPO_DIR && .venv/bin/python -m dimos.robot.cli.dimos --viewer none --robot-ip <ROBOT_IP> run sourccey-basic"
