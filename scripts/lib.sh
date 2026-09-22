# The experiment harness, sourced by every experiment's justfile.
#
# The orchestrator builds and collects; a target measures, in its own scratch.
# An experiment writes results/<exp>/<target>/{meta.json,<exp>.csv,raw/}.

set -uo pipefail

EXP_ROOT="${EXP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GUEST_DIR="$EXP_ROOT/miniosv"
HOST_DIR="$EXP_ROOT/llama.cpp-rknn"
MODELS_DIR="${MODELS_DIR:-$EXP_ROOT/models}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOCAL_TMP="${TMPDIR:-/tmp}/lros-orch"
mkdir -p "$LOCAL_TMP"

source "$(dirname "${BASH_SOURCE[0]}")/targets.sh"

exp_say()  { printf '\033[1m==>\033[0m %s\n' "$*" >&2; }
exp_warn() { printf '\033[33m==> %s\033[0m\n' "$*" >&2; }
exp_die()  { printf '\033[31m==> %s\033[0m\n' "$*" >&2; exit 1; }

# --- the target -------------------------------------------------------------

exp_target() {
    TARGET="$1"
    [ -n "${TGT_HOST[$TARGET]+x}" ] || exp_die "unknown target '$TARGET' (have: $(exp_targets))"
    T_HOST="${TGT_HOST[$TARGET]}"
    T_SCRATCH="${TGT_SCRATCH[$TARGET]}"
    T_PLATFORM="${TGT_PLATFORM[$TARGET]}"
    T_CORES="${TGT_CORES[$TARGET]}"
    T_MEM="${TGT_MEM[$TARGET]}"
    T_VCPUS="${TGT_VCPUS[$TARGET]}"
    T_ACCEL="${TGT_HOST_ACCEL[$TARGET]}"
    T_HOST_SRC="${TGT_HOST_SRC[$TARGET]}"
    T_REPO="${TGT_REPO[$TARGET]}"
    T_BUILDER="${TGT_BUILDER[$TARGET]}"
    exp_say "target $TARGET (${T_HOST:-here}), platform $T_PLATFORM, scratch $T_SCRATCH"
}

# Runs a shell command on the target, with $SCRATCH set.
tgt_sh() {
    if [ -z "$T_HOST" ]; then
        SCRATCH="$T_SCRATCH" bash -c "$1"
    else
        ssh -o BatchMode=yes "$T_HOST" "SCRATCH='$T_SCRATCH'; $1"
    fi
}

# Copies a local file or tree into the target's scratch.
tgt_push() {
    local src="$1" dst="$T_SCRATCH/$2"
    if [ -z "$T_HOST" ]; then
        mkdir -p "$(dirname "$dst")" && cp -r "$src" "$dst"
    else
        ssh -o BatchMode=yes "$T_HOST" "mkdir -p '$(dirname "$dst")'"
        rsync -a --info=none "$src" "$T_HOST:$dst" 2>/dev/null || scp -q -r "$src" "$T_HOST:$dst"
    fi
}

# The scratch tree, the guest tools from miniosv/scripts, and guest.sh: a
# launcher carrying the target's vAccel environment, dropped for CPU guests.
exp_setup() {
    exp_say "preparing $T_SCRATCH on $TARGET"
    tgt_sh "mkdir -p '$T_SCRATCH'/{bin,img,models,data,stage,run}" \
        || exp_die "cannot create the scratch tree on $TARGET"
    local f
    for f in run.py setargs.py imgedit.py mkdata.sh; do
        [ -f "$GUEST_DIR/scripts/$f" ] && tgt_push "$GUEST_DIR/scripts/$f" "bin/$f"
    done
    cat > "$LOCAL_TMP/guest-$TARGET.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
SCRATCH="$T_SCRATCH"
${TGT_VACCEL_ENV[$TARGET]}
if [ "\${LROS_VACCEL:-0}" != 1 ]; then
    unset VACCEL_PLUGINS VACCEL_BACKENDS
fi
: > "\$SCRATCH/run/AAVMF_VARS.fd"
export AAVMF_VARS="\$SCRATCH/run/AAVMF_VARS.fd"
exec "\$@"
EOF
    tgt_push "$LOCAL_TMP/guest-$TARGET.sh" "bin/guest.sh"
    tgt_sh "chmod +x '$T_SCRATCH/bin/guest.sh'"
    exp_has_accel || [ -z "${TGT_VACCEL_ENV[$TARGET]}" ] \
        || exp_warn "$TARGET is missing accelerator paths; vAccel arms will not run"
}

# Puts the target's models in its scratch: moved from where the target keeps
# them, copied if that is shared, shipped from here otherwise.
exp_sync_models() {
    local m src
    for m in ${TGT_MODELS[$TARGET]}; do
        [ "$(tgt_sh "[ -f '$T_SCRATCH/models/$m' ] && echo y")" = y ] && continue
        src="${TGT_MODEL_SRC[$TARGET]}/$m"
        if [ "$(tgt_sh "[ -f '$src' ] && echo y")" = y ]; then
            local op=mv; [ "$(tgt_sh "stat -f -c %T '$src'")" = nfs ] && op="cp -f"
            exp_say "$op $m into the scratch"
            tgt_sh "$op '$src' '$T_SCRATCH/models/$m'" || exp_die "could not $op $m"
        elif [ -f "$MODELS_DIR/$m" ]; then
            exp_say "shipping $m to $TARGET"
            tgt_push "$MODELS_DIR/$m" "models/$m"
        else
            exp_warn "$m is on neither $TARGET nor here; arms using it will fail"
        fi
    done
}

exp_model() { echo "$T_SCRATCH/models/$1"; }

exp_ncores() {
    local n=0 part IFS=,
    for part in $T_CORES; do
        case "$part" in
            *-*) n=$((n + ${part##*-} - ${part%%-*} + 1)) ;;
            *)   n=$((n + 1)) ;;
        esac
    done
    echo "$n"
}

# Whether the target's QEMU, plugin and firmware are all in place.
exp_has_accel() {
    [ -n "${TGT_VACCEL_ENV[$TARGET]}" ] || return 1
    [ -z "$(tgt_sh "${TGT_VACCEL_ENV[$TARGET]}
        for p in \"\$QEMU_VACCEL\" \"\$VACCEL_PLUGINS\" \"\$AAVMF_CODE\"; do
            [ -e \"\$p\" ] || echo missing
        done" 2>/dev/null)" ]
}

# --- one run per target -----------------------------------------------------

exp_lock() {
    if ! tgt_sh "mkdir '$T_SCRATCH/.lock' 2>/dev/null"; then
        exp_die "$TARGET is busy (held by $(tgt_sh "cat '$T_SCRATCH/.lock/owner'" 2>/dev/null)); 'just boards::unlock $TARGET' if stale"
    fi
    tgt_sh "echo '$(hostname):$$ $(date -Is)' > '$T_SCRATCH/.lock/owner'"
    trap exp_unlock EXIT INT TERM
}

exp_unlock() {
    exp_stop
    [ "${EXP_SWAP:-off}" = off ] && exp_swap on
    [ "$EXP_SWAP_EVENTS" -gt 0 ] && exp_warn "$EXP_SWAP_EVENTS arm(s) swapped; those rows are not comparable"
    tgt_sh "rm -rf '$T_SCRATCH/.lock'" 2>/dev/null || true
}

# Everything this harness starts on a target; bracketed so as not to match ssh.
exp_stop() {
    tgt_sh "pkill -9 -f '[r]un[.]py --arch' 2>/dev/null; \
            pkill -9 -f '[q]emu-system-aarch64 -m' 2>/dev/null; \
            pkill -9 -f '[l]lama-batched-bench' 2>/dev/null; \
            pkill -9 -f '[l]lama-server' 2>/dev/null; true"
}

# --- memory state between arms ----------------------------------------------

# Drops the page cache and empties swap, unless EXP_NO_DROP keeps it warm.
exp_flush_memory() {
    [ -n "${EXP_NO_DROP:-}" ] && return 0
    tgt_sh "sync
        drop() { echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'; }
        drop 2>/dev/null || exit 0
        used=\$(awk '/SwapTotal/{t=\$2} /SwapFree/{f=\$2} END{print t-f}' /proc/meminfo)
        [ \"\$used\" -gt 0 ] && sudo -n swapoff -a 2>/dev/null && sudo -n swapon -a 2>/dev/null
        true" >/dev/null 2>&1
}

# Swap is turned off for a run and the devices that were on turned back on by
# name: the orangepi's zram is not in fstab, so swapon -a would not restore it.
EXP_SWAP_DEVS=""
exp_swap() {
    case "$1" in
        off) EXP_SWAP_DEVS=$(tgt_sh "awk 'NR>1{printf \"%s \", \$1}' /proc/swaps" 2>/dev/null)
             tgt_sh "sudo -n swapoff -a 2>/dev/null" && exp_say "swap off on $TARGET" \
                 || exp_warn "could not turn swap off on $TARGET" ;;
        on)  [ -z "$EXP_SWAP_DEVS" ] && return 0
             tgt_sh "sudo -n swapon $EXP_SWAP_DEVS 2>/dev/null || sudo -n swapon -a 2>/dev/null" \
                 && exp_say "swap restored on $TARGET" || exp_warn "could not restore swap: $EXP_SWAP_DEVS" ;;
    esac
}

exp_swap_used() {
    tgt_sh "awk '/SwapTotal/{t=\$2} /SwapFree/{f=\$2} END{print t-f}' /proc/meminfo" 2>/dev/null
}

# An arm during which swap grew measured the swap path; counted and reported.
EXP_SWAP_EVENTS=0
exp_swap_check() {
    local before="$1" after; after=$(exp_swap_used)
    if [ -n "$before" ] && [ -n "$after" ] && [ "$after" -gt "$before" ]; then
        EXP_SWAP_EVENTS=$((EXP_SWAP_EVENTS + 1))
        exp_warn "swap grew by $((after - before)) kB during this arm"
    fi
}

# --- results ----------------------------------------------------------------

# exp_begin <name> <csv header>: the result directory, meta.json and the lock.
exp_begin() {
    EXP_NAME="$1"
    EXP_DIR="$RESULTS_DIR/$EXP_NAME/$TARGET"
    EXP_RAW="$EXP_DIR/raw"
    EXP_CSV="$EXP_DIR/$EXP_NAME.csv"
    mkdir -p "$EXP_RAW"
    printf '%s\n' "$2" > "$EXP_CSV"
    cat > "$EXP_DIR/meta.json" <<EOF
{
  "experiment": "$EXP_NAME",
  "target": "$TARGET",
  "platform": "$T_PLATFORM",
  "date": "$(date -Is)",
  "cores": "$T_CORES",
  "guest_mem": "$T_MEM",
  "guest_vcpus": $T_VCPUS,
  "reps": $EXP_REPS,
  "guest_rev": "$(git -C "$GUEST_DIR" rev-parse --short HEAD 2>/dev/null || echo none)",
  "guest_dirty": $(git -C "$GUEST_DIR" diff --quiet 2>/dev/null && echo false || echo true),
  "host_rev": "$(git -C "$HOST_DIR" rev-parse --short HEAD 2>/dev/null || echo none)",
  "swap": "$(tgt_sh "swapon --show=NAME,TYPE,SIZE --noheadings | tr '\\n' ';'" 2>/dev/null)",
  "orchestrator": "$(hostname)"
}
EOF
    exp_lock
    [ "${EXP_SWAP:-off}" = off ] && exp_swap off
    exp_flush_memory
    exp_say "$EXP_NAME -> $EXP_DIR"
}

exp_row() { local IFS=,; printf '%s\n' "$*" >> "$EXP_CSV"; EXP_ROWS=$((${EXP_ROWS:-0} + 1)); }

exp_finish() {
    if [ "${EXP_ROWS:-0}" -eq 0 ]; then
        exp_warn "$EXP_NAME on $TARGET produced NO rows; see $EXP_RAW"
        return 1
    fi
    exp_say "$EXP_NAME on $TARGET: ${EXP_ROWS} row(s) -> $EXP_CSV"
}

EXP_REPS="${EXP_REPS:-3}"
exp_reps() { seq 1 "$EXP_REPS"; }

exp_parse() { python3 "$EXP_ROOT/scripts/parse.py" "$@"; }

# --- building ---------------------------------------------------------------

# exp_build_guest <variant> [vaccel] [sgemm]: a guest image built here, one out
# directory per variant; viai selects the graph-level backend. Echoes the image.
exp_build_guest() {
    local variant="$1" vaccel="${2:-1}" sgemm="${3:-1}" vgpu=0
    local out="build/$variant.aarch64"
    [ "$variant" = viai ] && { vgpu=1; vaccel=0; }
    exp_say "build guest $variant (vaccel=$vaccel viai=$vgpu sgemm=$sgemm)"
    (
        flock 9
        nix develop "$GUEST_DIR" --command make -C "$GUEST_DIR" -j"$(nproc)" \
            app=app/llama.cpp arch=aarch64 out="$out" \
            conf_vaccel="$vaccel" conf_viai="$vgpu" conf_llama_sgemm="$sgemm" \
            > "$LOCAL_TMP/build-$variant.log" 2>&1
    ) 9> "$LOCAL_TMP/build-$variant.lock" \
        || { tail -30 "$LOCAL_TMP/build-$variant.log" >&2; exp_die "guest build $variant failed"; }
    echo "$GUEST_DIR/$out/loader.img"
}

# Ships an image unless the target has the same content. Echoes its path there.
exp_ship_image() {
    local variant="$1" img="$2" have
    have=$(tgt_sh "md5sum '$T_SCRATCH/img/$variant.img' 2>/dev/null | cut -d' ' -f1")
    if [ "$have" != "$(md5sum "$img" | cut -d' ' -f1)" ]; then
        exp_say "shipping $variant image to $TARGET"
        tgt_push "$img" "img/$variant.img"
    fi
    echo "$T_SCRATCH/img/$variant.img"
}

# exp_build_host <preset> [targets]: a native llama.cpp build on the target,
# against its own libraries. Echoes the bin directory.
exp_build_host() {
    local preset="$1" targets="${2:-llama-batched-bench llama-cli}" flags=""
    case "$preset" in
        cpu)    ;;
        rknnoh) flags="-DGGML_RKNNOH=ON" ;;
        cuda)   flags="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=\${CUDA_ARCH:-native}" ;;
        *)      exp_die "unknown host preset $preset" ;;
    esac
    exp_say "build host $preset on $TARGET"
    tgt_sh "${TGT_BUILD_ENV[$TARGET]}
        set -e
        dir='$T_HOST_SRC/build-$preset'
        if [ -f \"\$dir/CMakeCache.txt\" ] && [ ! -f \"\$dir/Makefile\" ]; then rm -rf \"\$dir\"; fi
        [ -f \"\$dir/CMakeCache.txt\" ] || cmake -S '$T_HOST_SRC' -B \"\$dir\" -DCMAKE_BUILD_TYPE=Release $flags
        cmake --build \"\$dir\" -j\$(nproc) -t $targets
    " > "$LOCAL_TMP/build-host-$preset.log" 2>&1 \
        || { tail -30 "$LOCAL_TMP/build-host-$preset.log" >&2; exp_warn "host build $preset failed on $TARGET"; return 1; }
    echo "$T_HOST_SRC/build-$preset/bin"
}

# exp_mkdata <staging dir> <name> [size]: an ext4 image of the staging tree,
# which the guest mounts at /root. Echoes its path on the target.
exp_mkdata() {
    local stage="$1" name="$2" size="${3:-2G}"
    exp_say "data image $name"
    tgt_sh "rm -rf '$T_SCRATCH/stage/$name'"
    tgt_push "$stage/" "stage/$name"
    tgt_sh "MKDATA_ASSUME_YES=1 bash '$T_SCRATCH/bin/mkdata.sh' \
        '$T_SCRATCH/data/$name.img' '$T_SCRATCH/stage/$name' $size" \
        > "$LOCAL_TMP/mkdata-$name.log" 2>&1 \
        || { tail -20 "$LOCAL_TMP/mkdata-$name.log" >&2; exp_die "mkdata $name failed"; }
    echo "$T_SCRATCH/data/$name.img"
}

# --- the accelerator stack --------------------------------------------------

# The group cache, when it answers; checked once per run.
exp_cache_opt() {
    if [ -z "${EXP_CACHE_OK+x}" ]; then
        curl -sS -m 8 -o /dev/null "$NIX_CACHE/nix-cache-info" 2>/dev/null && EXP_CACHE_OK=1 || EXP_CACHE_OK=0
    fi
    [ "$EXP_CACHE_OK" = 1 ] && printf -- "--option extra-substituters %s" "$NIX_CACHE"
}

# exp_nix_build <attr>...: built on the target's builder and copied over.
# Echoes the out paths.
exp_nix_build() {
    local attrs="" a out
    for a in "$@"; do attrs="$attrs '$T_REPO#${a#.#}'"; done
    local cmd="nix build --no-link --print-out-paths $(exp_cache_opt) $attrs"
    if [ -z "$T_BUILDER" ]; then
        tgt_sh "$cmd" || { exp_warn "nix build failed on $TARGET"; return 1; }
        return 0
    fi
    out=$(ssh -o BatchMode=yes "$T_BUILDER" "$cmd") || { exp_warn "nix build failed on $T_BUILDER"; return 1; }
    ssh -o BatchMode=yes "$T_BUILDER" "nix copy --to 'ssh://$T_HOST' $(printf "'%s' " $out)" \
        || { exp_warn "could not copy the closure to $TARGET"; return 1; }
    printf '%s\n' "$out"
}

# A store path linked into the scratch as a GC root, so collection spares it.
exp_pin() {
    tgt_sh "nix-store --realise '$1' --add-root '$2' --indirect >/dev/null" \
        || tgt_sh "ln -sfn '$1' '$2'"
}

# The plugin, built on the target by vaccel_plugins/build.sh from sources
# synced there first when its checkout is not this one.
exp_accel_plugin() {
    if [ "$T_REPO" != "$EXP_ROOT" ] && [ -n "$T_HOST" ]; then
        local d
        for d in vaccel_plugins miniosv/app/llama.cpp/ggml/src/ggml-viai; do
            rsync -a --info=none "$EXP_ROOT/$d/" "$T_HOST:$T_REPO/$d/" || exp_die "could not sync $d"
        done
        rsync -a --info=none "$EXP_ROOT/flake.nix" "$EXP_ROOT/flake.lock" "$T_HOST:$T_REPO/"
    fi
    exp_say "build plugin on $TARGET ($T_PLATFORM)"
    tgt_sh "SCRATCH='$T_SCRATCH' '$T_REPO/vaccel_plugins/build.sh' '$T_PLATFORM'" \
        > "$LOCAL_TMP/accel-plugin.log" 2>&1 \
        || { tail -25 "$LOCAL_TMP/accel-plugin.log" >&2; exp_die "plugin build failed on $TARGET"; }
}

# QEMU with vAccel and the plugin, unless both are already in place.
exp_accel_setup() {
    exp_has_accel && { exp_say "accelerator stack already in $T_SCRATCH"; return 0; }
    local qemu; qemu=$(exp_nix_build ".#qemu-vaccel") || return 1
    tgt_sh "mkdir -p '$T_SCRATCH/plugin'"
    exp_pin "$qemu" "$T_SCRATCH/qemu-vaccel"
    exp_accel_plugin
    exp_setup
}

# --- running ----------------------------------------------------------------

# exp_guest <arm> <rep> <image> <args> [nvme]...: one guest run, on a copy of
# the image since run.py rewrites it. The console is left in EXP_LOG; EXP_VACCEL,
# EXP_QEMU, EXP_GUEST_ENV and EXP_TIMEOUT adjust the run.
exp_guest() {
    local arm="$1" rep="$2" image="$3" args="$4"; shift 4
    local log="$EXP_RAW/$arm.$rep.log" run_img="$T_SCRATCH/run/$EXP_NAME-$arm.img" nvme="" d
    for d in "$@"; do nvme="$nvme --emulated-nvme '$d'"; done
    exp_stop
    exp_flush_memory
    [ -z "${EXP_QEMU:-}" ] && [ -n "${TGT_VACCEL_ENV[$TARGET]}" ] \
        && EXP_QEMU=$(tgt_sh "${TGT_VACCEL_ENV[$TARGET]}
            echo \"\$QEMU_VACCEL\"" 2>/dev/null | tail -1)
    local swap_before; swap_before=$(exp_swap_used)
    exp_say "guest $arm rep $rep"
    tgt_sh "cp -f '$image' '$run_img'
        export LROS_VACCEL=${EXP_VACCEL:+1} ${EXP_GUEST_ENV:-}
        '$T_SCRATCH/bin/guest.sh' taskset -c $T_CORES timeout ${EXP_TIMEOUT:-900} \
            python3 '$T_SCRATCH/bin/run.py' --arch aarch64 ${EXP_VACCEL:+--vaccel} \
              ${EXP_QEMU:+--qemu-path '$EXP_QEMU'} -m ${EXP_MEM:-$T_MEM} -c ${EXP_VCPUS:-$T_VCPUS} \
              --image-path '$run_img' $nvme --args \"$args\" < /dev/null" > "$log" 2>&1
    local rc=$?
    exp_swap_check "$swap_before"
    [ $rc -ne 0 ] && exp_warn "$arm rep $rep exited $rc (see $log)"
    EXP_LOG="$log"
}

# exp_guest_retry: exp_guest, killed and retried when the UEFI stub hangs
# before the kernel starts (BOOT_TIMEOUT seconds); a booted run is never retried.
exp_guest_retry() {
    local arm="$1" rep="$2" attempt pid waited log="$EXP_RAW/$1.$2.log"
    for attempt in 1 2 3; do
        rm -f "$log"
        exp_guest "$@" &
        pid=$! waited=0
        while kill -0 "$pid" 2>/dev/null && ! grep -qa "Booted up in" "$log" 2>/dev/null; do
            if [ "$waited" -ge "${BOOT_TIMEOUT:-90}" ]; then
                exp_warn "$arm rep $rep attempt $attempt stuck before boot; killing it"
                tgt_sh "pkill -KILL -f 'qemu-system.*$EXP_NAME-$arm[.]img'"
                break
            fi
            sleep 5; waited=$((waited + 5))
        done
        wait "$pid"
        EXP_LOG="$log"
        grep -qa "Booted up in" "$log" && return
        exp_warn "$arm rep $rep attempt $attempt never booted; retrying"
    done
}

# exp_host <arm> <rep> <command>: one native run on the pinned cores, with
# EXP_HOST_ENV prepended. The console is left in EXP_LOG.
exp_host() {
    local arm="$1" rep="$2" cmd="$3" log="$EXP_RAW/$1.$2.log"
    exp_flush_memory
    local swap_before; swap_before=$(exp_swap_used)
    exp_say "host $arm rep $rep"
    tgt_sh "${EXP_HOST_ENV:-}
        taskset -c $T_CORES timeout ${EXP_TIMEOUT:-900} $cmd < /dev/null" > "$log" 2>&1
    local rc=$?
    exp_swap_check "$swap_before"
    [ $rc -ne 0 ] && exp_warn "$arm rep $rep exited $rc (see $log)"
    EXP_LOG="$log"
}
