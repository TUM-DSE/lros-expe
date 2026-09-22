#! /usr/bin/env nix
#! nix shell nixpkgs#patchelf --command bash

# Prints the command that runs a binary (relative to the repo root, which the
# VM mounts at /root) inside the VM, its rpath remapped to the 9p host store
# and its loader taken from the guest's system closure ($CONF).

bin=$1; shift

rpath=$(patchelf --print-rpath "$bin")
interp=$(patchelf --print-interpreter "$(readlink -f "$CONF/sw/bin/nix")")

IFS=':' read -ra rpath_split <<< "$rpath"
# Resolve $ORIGIN (loader-relative to the binary) ourselves: the guest shell
# would otherwise try to expand it as a variable.
rpath_split=("${rpath_split[@]/#\$ORIGIN//root/$(dirname "$bin")}")
rpath=$(printf ":%s" "${rpath_split[@]/#\/nix\/store/\/nix\/.ro-store-vmux}")
rpath=${rpath:1}

echo "LD_LIBRARY_PATH=$rpath $interp /root/$bin $*"