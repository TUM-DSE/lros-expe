#!/usr/bin/env bash
# Builds the plugin on the board you are logged into, without the experiment
# harness: the same build-plugin.sh that `just exp::accel-plugin` drives over
# ssh, run inside this repo's plugin devshell so the paths and the toolchain
# come from flake.lock rather than from anything written down here.
#
#   vaccel_plugins/build.sh [rk3588|orin]     platform, else autodetected
#
# Writes $SCRATCH/plugin; point VACCEL_PLUGINS and VACCEL_BACKENDS at the .so to
# use it.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

PLATFORM="${1:-}"
if [ -z "$PLATFORM" ]; then
    compat=$(tr -d '\0' < /proc/device-tree/compatible 2>/dev/null || true)
    case "$compat" in
        *rk3588*) PLATFORM=rk3588 ;;
        *tegra*)  PLATFORM=orin ;;
        *) echo "cannot tell what board this is; pass rk3588 or orin" >&2; exit 1 ;;
    esac
fi
case "$PLATFORM" in
    rk3588) SCRATCH="${SCRATCH:-/var/tmp/lros}" ;;
    orin)   SCRATCH="${SCRATCH:-/scratch/$USER/lros}" ;;
    *) echo "no plugin for platform '$PLATFORM'" >&2; exit 1 ;;
esac
export REPO SCRATCH

# The orangepi is Ubuntu, whose nix-bin package owns /usr/bin/nix at 2.6 -- old
# enough that it cannot parse this flake's inputs at all. Its daemon and default
# profile are 2.31, so prefer those wherever they exist.
NIX=nix
for c in /nix/var/nix/profiles/default/bin/nix "$HOME/.nix-profile/bin/nix"; do
    [ -x "$c" ] && { NIX="$c"; break; }
done

exec "$NIX" develop "$REPO#plugin-$PLATFORM" \
    --command "$REPO/vaccel_plugins/build-plugin.sh" all
