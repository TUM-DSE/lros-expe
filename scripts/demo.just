cli:
	just -f {{justfile()}} run_lros_stdout \
	{{LROS_PATH}}/.unikraft/build-cli/cli_qemu-arm64 \
	8G "llama-cli -m /root/models/Llama-3.2-1B-Instruct-f16.gguf --no-mmap --no-perf"