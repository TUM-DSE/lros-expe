# The boards the experiments run on: setup, state and the accelerator stack.

HERE := source_directory()
LIB := HERE / "lib.sh"

[doc('the targets and whether they answer')]
targets:
	#!/usr/bin/env bash
	source {{LIB}}
	for t in $(exp_targets); do
	  host="${TGT_HOST[$t]}" state=local
	  [ -n "$host" ] && { ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" true 2>/dev/null && state=up || state=UNREACHABLE; }
	  printf '%-10s %-34s %-8s %s\n' "$t" "${host:-(here)}" "${TGT_PLATFORM[$t]}" "$state"
	done

[doc('scratch tree, launcher and models on a target')]
setup target="orangepi":
	#!/usr/bin/env bash
	source {{LIB}}; exp_target {{target}}
	exp_setup
	exp_sync_models

[doc('what a target scratch holds')]
status target="orangepi":
	#!/usr/bin/env bash
	source {{LIB}}; exp_target {{target}}
	tgt_sh 'df -h "$SCRATCH" | tail -1; du -sh "$SCRATCH"/* 2>/dev/null | sort -rh'

[doc('release a stale lock and stop what it left running')]
unlock target="orangepi":
	#!/usr/bin/env bash
	source {{LIB}}; exp_target {{target}}
	exp_stop
	tgt_sh 'rm -rf "$SCRATCH/.lock"'

[doc('QEMU with vAccel and the plugin, into the target scratch')]
accel-setup target="orangepi":
	#!/usr/bin/env bash
	source {{LIB}}; exp_target {{target}}
	exp_setup
	exp_accel_setup

[doc('rebuild the vAccel plugin only')]
accel-plugin target="joy":
	#!/usr/bin/env bash
	source {{LIB}}; exp_target {{target}}
	exp_setup
	exp_accel_plugin
