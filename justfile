import "scripts/deps.just"
import "scripts/vm.just"
import "scripts/build.just"
import "scripts/plot.just"
import "scripts/bench.just"

proot := justfile_directory()

models_dir := proot+"/models"
models_to_get := "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-f16.gguf"

default:
    @just --choose

help:
    just --list

get_models:
  #!/usr/bin/env bash
  mkdir -p {{models_dir}}
  cd {{models_dir}}
  declare -A models={{models_to_get}}
  for url in $models; do
    echo $url
    wget -nc -q --show-progress "$url"
  done
