# lros-expe
Experiments and benchmarks for the LROS project

## Hardware details

On the Orange Pi Ultra that we have, cores [0-3] are Cortex-A55 (in-order -> more efficient) while cores [4-7] are Cortex-A76 (O3 -> more performant)

## Setup

Clone with `git clone [url] --recursive` or execute `git submodule update --init --recursive` to initialize the submodules 

To get the models, run the `scripts/download_models.sh` script.

## Build

1. vaccel
meson >= 1.1,

```
cd vaccel
cd scripts/common; git apply ../../submodules.patch; cd ../..
meson setup --buildtype=release build
meson compile -C build
meson install -C build --destdir=out
sed -i "s/prefix=/prefix=\/home\/$(whoami)\/lros-expe\/vaccel\/build\/out/" /home/$(whoami)/lros-expe/vaccel/build/out/usr/local/lib/pkgconfig/vaccel.pc
```

2. lros-qemu
Need: python3-tomli, libglib2.0-dev

```
cd lros-qemu
mkdir build
cd build
CFLAGS=-Wno-error PKG_CONFIG_PATH=/home/$(whoami)/lros-expe/vaccel/build/out/usr/local/lib/aarch64-linux-gnu/pkgconfig ../configure --target-list=aarch64-softmmu --enable-virtfs
make -j
```

3. llama.cpp

```
cmake -B build
cmake --build build --config Release -j
```


## Using LoRA adapters with the models

1. Find some suitable adapters for the model you want to infer: For example for the Llama-3.1-1b-Instruct model you can use [Llama-TOS](https://huggingface.co/CodeHima/Llama_TOS) and [MentalChat-16K](https://huggingface.co/khazarai/MentalChat-16K).
2. Clone the repo containg the adapter_config.json and adapter_model.safetensors files.
3. Convert the LoRA into GGUF format using the convert_lora_to_gguf.py script from the llama.cpp repo
   1. Install the requirements using `pip install -r requirements/requirements-convert_lora_to_gguf.txt`
   2. `./convert_lora_to_gguf.py --outfile <lora-name>.gguf --outtype f16 <cloned lora repo>`
4. Start llama-server with the LoRAs: Add `--lora-scaled path/to/lora.gguf 0` for every LoRA you want to supply.
   1. Note: It should be possible to just do `--lora path/to/lora.gguf` and additionally add `--lora-init-without-apply` but that did not work in my tests
5. Modify the applied LoRA(s) using:
   1. A `POST` request to `/lora-adapters` supplying `[{"id": 0, "scale": 0.2},{"id": 1, "scale": 0.8}]` as the request body (not included LoRAs are automatically scaled to 0).
   2. Per request by adding a `lora` parameter to the json request body, that contains an array like above