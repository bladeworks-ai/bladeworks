#!/usr/bin/env bash
# Supported Linux installer for Bladeworks.
#
# Native packages are installed system-wide. Bladeworks itself is isolated in
# ~/.local/share/bladeworks/venv and only its launcher is exposed on PATH.

set -euo pipefail

bladeworks_spec="${BLADEWORKS_SPEC:-bladeworks}"
install_root="${BLADEWORKS_INSTALL_ROOT:-${XDG_DATA_HOME:-${HOME}/.local/share}/bladeworks}"
bin_dir="${BLADEWORKS_BIN_DIR:-${HOME}/.local/bin}"
python_command="python3"

if [[ ! -r /etc/os-release ]]; then
  echo "error: /etc/os-release is missing; this Linux distribution is unsupported" >&2
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release

run_privileged() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
    return
  fi
  if ! command -v sudo >/dev/null 2>&1; then
    echo "error: this installer needs root access, but sudo is unavailable" >&2
    exit 1
  fi
  sudo "$@"
}

install_apt_packages() {
  run_privileged apt-get update
  run_privileged apt-get install -y ffmpeg fonts-dejavu-core libraqm0 libfribidi0 libharfbuzz0b python3 python3-venv

  if ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
    echo "error: ${ID} ${VERSION_ID:-unknown} provides Python $(python3 --version 2>&1), but Bladeworks requires Python 3.10 or newer." >&2
    echo "Use Ubuntu 22.04 or newer, or Debian 12 or newer." >&2
    exit 1
  fi
}

install_dnf_packages() {
  if ! dnf list --available ffmpeg >/dev/null 2>&1 && ! rpm -q ffmpeg >/dev/null 2>&1; then
    echo "error: ffmpeg is unavailable in enabled repositories." >&2
    echo "Enable RPM Fusion for ${ID} ${VERSION_ID}, then rerun this installer." >&2
    exit 1
  fi
  run_privileged dnf install -y dejavu-sans-fonts ffmpeg fribidi harfbuzz libraqm python3
}

install_rhel_packages() {
  if ! dnf list --available python3.11 >/dev/null 2>&1 && ! rpm -q python3.11 >/dev/null 2>&1; then
    echo "error: Python 3.11 is unavailable in enabled repositories." >&2
    echo "Bladeworks supports RHEL-family releases with the python3.11 package." >&2
    exit 1
  fi
  install_dnf_packages
  run_privileged dnf install -y python3.11
  python_command="python3.11"
}

case "${ID}" in
  ubuntu|debian)
    install_apt_packages
    ;;
  fedora)
    install_dnf_packages
    ;;
  rhel|rocky|almalinux)
    install_rhel_packages
    ;;
  *)
    echo "error: unsupported Linux distribution: ${ID} ${VERSION_ID:-unknown}" >&2
    echo "Supported families: Debian/Ubuntu and Fedora/RHEL." >&2
    exit 1
    ;;
esac

"${python_command}" -m venv "${install_root}/venv"
"${install_root}/venv/bin/python" -m pip install --upgrade pip
"${install_root}/venv/bin/python" -m pip install "${bladeworks_spec}"
mkdir -p "${bin_dir}"
ln -sfn "${install_root}/venv/bin/bladeworks" "${bin_dir}/bladeworks"

if [[ ":${PATH}:" != *":${bin_dir}:"* ]]; then
  echo "note: add ${bin_dir} to PATH to invoke bladeworks directly" >&2
fi

"${install_root}/venv/bin/bladeworks" doctor
echo "Bladeworks installed at ${install_root}/venv"
