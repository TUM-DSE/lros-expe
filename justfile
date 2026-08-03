import "scripts/deps.just"
import "scripts/vm.just"
import "scripts/build.just"
import "scripts/plot.just"
import "scripts/bench.just"
import "scripts/demo.just"

proot := justfile_directory()

models_dir := proot+"/models"
models_to_get := "
	https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-f16.gguf
	https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf
	https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-fp16.gguf
	https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf
	https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-f16.gguf
	https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q8_0.gguf
"

default:
    @just --choose

help:
    just --list

get_models:
  #!/usr/bin/env bash
  mkdir -p {{models_dir}}
  cd {{models_dir}}
  models="{{models_to_get}}"
  for url in $models; do
    echo $url
    wget -nc -q --show-progress "$url"
  done
