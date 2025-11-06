import? "../justfile"
out_dir := proot+"/../out"
models_dir := "/root/models"

cores := "1 4"
mem_size := "1G 4G"
model := "Llama-3.2-1B-Instruct-Q3_K_L.gguf"
prompt := "Describe a combustion engine"
nb_tokens_to_predict := "64"

kvcache:
	#!/usr/bin/env bash
	
	for c in {{cores}}
	do
		for m in {{mem_size}}
		do
			echo "$c $m"
			just vm $c $m > /dev/null 2> /dev/null &
			sleep 40
			
			just ssh_local "echo 1 > /proc/sys/vm/drop_caches; /root/llama.cpp/build/bin/llama-cli -m {{models_dir}}/{{model}} -p {{prompt}} -t $c -n '{{nb_tokens_to_predict}}' -no-cnv" > {{out_dir}}/raw_"$c"_"$m".txt 
			sleep 10
			just ssh_local "echo 1 > /proc/sys/vm/drop_caches; perf record -F 99 -o {{out_dir}}/perf_$c_$m.data -ag -- /root/llama.cpp/build/bin/llama-cli -m {{models_dir}}/{{model}} -p '{{prompt}}' -t $c -n {{nb_tokens_to_predict}} -no-cnv"
			sleep 10
			just ssh_local "poweroff"
		done
	done
