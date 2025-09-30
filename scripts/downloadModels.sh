#!/usr/bin/env bash

models_dir="../models"

URL_TO_DL=(
 ["Llama-3.2-1B-Instruct-f16"] = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-f16.gguf"
)


dl_model() {
	local name="$1"
	local url="$2"
	echo $name

	if [ -f "$models_dir/$name" ]; then
		echo "$name already exists, skipping."
	else
		wget -q --show-progress -O "$models_dir/$name" "$url"
	fi
}

mkdir -p $models_dir
for filename in "${!URL_TO_DL[@]}"; do
	echo $filename
	#dl_model $filename "${URL_TO_DL[$filename]}"
done
