#!/usr/bin/env bash
set -euo pipefail
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
for arg in "$@"; do
    if [[ "$arg" == --dry-run || "$arg" == --help ]]; then
        if command -v python3 >/dev/null; then
            exec python3 "$script_dir/install_redshift.py" "$@"
        fi
        echo 'Bootstrap requires Python 3; normal installation installs it on Ubuntu 24.04.'
        exit 0
    fi
done
if [[ $(id -u) != 0 ]]; then
    exec sudo bash "$0" "$@"
fi
if ! command -v python3 >/dev/null; then
    for arg in "$@"; do
        [[ "$arg" != --offline ]] || { echo 'Offline installation requires Python 3.' >&2; exit 1; }
    done
    [[ -r /etc/os-release ]] || { echo 'Ubuntu 24.04 is required.' >&2; exit 1; }
    . /etc/os-release
    [[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]] || exit 1
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3
fi
exec python3 "$script_dir/install_redshift.py" "$@"
