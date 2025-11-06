#! /usr/bin/env nix
#! nix shell nixpkgs#patchelf --command bash

rpath=$(patchelf --print-rpath /root/lros/llama.cpp-rknn/build/bin/llama-batched-bench)
interp=$(patchelf --print-interpreter "$(which patchelf)")

IFS=':' read -ra rpath_split <<< "$rpath"
rpath=$(printf ":%s" "${rpath_split[@]/#\/nix\/store/\/nix\/.ro-store-vmux}")
rpath=${rpath:1}

LD_LIBRARY_PATH="$rpath" $interp "$@"
