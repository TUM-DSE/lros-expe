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

```bash
cd vaccel
cd scripts/common; git apply ../../submodules.patch; cd ../..
meson setup --buildtype=release build
meson compile -C build
meson install -C build --destdir=out
sed -i "s/prefix=/prefix=\/home\/$(whoami)\/lros-expe\/vaccel\/build\/out/" /home/$(whoami)/lros-expe/vaccel/build/out/usr/local/lib/pkgconfig/vaccel.pc
```

2. lros-qemu
Need: python3-tomli, libglib2.0-dev

```bash
cd lros-qemu
mkdir build
cd build
CFLAGS=-Wno-error PKG_CONFIG_PATH=/home/$(whoami)/lros-expe/vaccel/build/out/usr/local/lib/aarch64-linux-gnu/pkgconfig ../configure --target-list=aarch64-softmmu --enable-virtfs
make -j
```

3. lros/llama.cpp-rknn

```bash
cmake -B build -DGGML_RKNN=ON
cmake --build build --config Release --target llama-batched-bench -j
```

### Unikernel

#### Requirements

- GCC 13 (I think neither higher nor lower work)
- Make
- git

Ideally (unless you want to manually download dependencies and build):
- Kraftkit


#### Configuration

You can use one of the pre configured configurations:

- bench-vaccel: Benchmark program as frontend, vAccel backend enabled
- bench-vaccel-novec: Benchmark program as frontend, vAccel backend enabled, vectorization disabled for CPU backend
- bench-cpu: Benchmark program as frontend, cpu backend only
- bench-cpu-novec: Benchmark program as frontend, cpu backend only, vectorization disabled for CPU backend

Alternatively you can create a configuration yourselves:

```bash
kraft menu -t qemu/arm64
```

In the menu you can select different options.
The default config provided in the Kraftfile already includes settings required to run llama.cpp.
Some important observations:
- "Device Drivers">"Random Number Generator">"CPU generated randomness" (CONFIG_LIBUKRANDOM_LCPU) must be off/unset or QEMU will not run the kernel. *It is automatically turned on on every reconfiguration.*
- "Device Drivers">"Interrupt controller">"Arm Generic Interrupt Controller (GICv3)" (CONFIG_LIBUKINTCTLR_GICV3) is required to run with QEMU. GICv2 is not enough. Should be turned on automatically.
- "Application Options">"Llama.cpp Application that runs in the unikernel"(LLAMACPP_APP_\*): Select Bench or CLI to select a frontend.
- "Application Options">"Use llamafile sgemm kernels"(LLAMACPP_LLAMAFILE_SGEMM): Select whether you want the vectorized matrix multiplication CPU kernels.
- "Application Options">"Use vaccel llama backend"(LLAMACPP_VACCEL): Enable the vaccel backend. Requires "Library Configuration">"vaccelrt: Vaccel runtime system library" (LIBVACCELRT)
- Default does not include a mounted filesystem: It is suggested to setup a ramfs and then mount your files into that with 9pfs. To set this up:
	- Enable "Library Configuration">"vfscore: VFS Core Interface">"Compiled-in filesystem table (up to 4 entries, earliest prio)" (LIBVFSCORE_AUTOMOUNT_CI)
	- Open the resulting submenu and setup Entry 0 as "Mount point"="/", "Filesystem"="ramfs", leave the other fields default
	- In the submenu "Entry options" enable "mkmp - Create mountpoint"
	- Do the same for Entry 1, with "Device"="rootfs", "Mount point"="/root", "Filesystem"="9pfs". Don't forget to set mkmp. You can substitute a different device name and moun point but other steps will assume these values.
- Some configurations get stuck in infinite loops during boot. For example in a previous iteration uklibparam was not needed by any of the selected features. Still disabling it caused an infinite loop, eventhough no added code was executed.

#### Build unikernel

```
kraft build --no-configure
```

Select the configuration you want (or qemu/arm64 if you manually created one).

This will automatically download library dependencies and then invoke make to configure and build the project.
The resulting kernel and intermediate files will be placed in .unikraft/build.

## Usage

To run the unikernel once it has been built, you want to use the following command:

```bash
qemu-system-aarch64 \
		-kernel <img> \
		-append "<args>" \
		-cpu host \
		-machine virt \
		-m size=<mem> \
		-smp cpus=1,threads=1,sockets=1 \
		-parallel none \
		-device virtio-9p-pci,fsdev=rootfs,mount_tag=rootfs \
		-fsdev local,id=rootfs,path=<path>,security_model=mapped-xattr \
		-display none \
		-nographic \
		-vga none \
		-no-reboot \
		-rtc base=utc \
		-enable-kvm \
```

Substitute according to your setup:
- `<img>`: Path to the built unikernel image
- `<mem>`: How much memory you want to assign (e.g. 2G for 2 Gigabyte)
- `<path>`: Path to the filesystem you want to mount in the unikernel (should contain the model etc.). Using the suggested configuration from above will automatically mount this in the unikernel in the path `/root`.

You can specify the command line options for llama.cpp using `<args>`. The following only list some options. For a full list please conult the llama.cpp documentation:

Some important options:
	- `-m <path>`: Path to the inference model to use. We tested using [Llama-3.2-1B-Instruct-f16.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-f16.gguf).
	- `-t 1`: Set to single thread execution. Unikraft only works with 1 thread so this is important.
	- `--no-mmap`: Disable model memory mapping (using the custom memory prefetcher) and use read instead. Needs big enough memory allocation to fit whole model.
	
Depending on the application you choose when building the image you can specify additional options:

1. Bench:
	- `-npp <c>`: Comma separated list of token counts to use for prompt processing phase of the bench. We test using "16,32".
	- `-ntg <c>`: Comma separated list of token counts to use for text generation phase of the bench. We test using "128,256".
	- `-npl <c>`: Comma separated list of batch sizes to use for the bench. We test using 1 and 4.
	
2. CLI:
	- `-no-cnv`: Disable conversation mode.
	- `--interactive`: Enable interactive mode.
	- `--no-warmup`: Skip model warmup.
	- `-p <p>`: Prompt to use.
	- `-n`: Number of tokens to use.
	
	
### vaccel

If you want to use the vAccel backend there are some additional steps to run the project.

1. Build/install vAccel. (see Build above)
2. Build/install the custom qemu (see Build above).
3. Build the vAccel plugins https://github.com/TUM-DSE/lros/tree/staging/vaccel_plugins.
4. Enable vAccel for the unikernel (see Configuration above).
5. Set the environment variable `VACCEL_PLUGINS` to the path of the vaccel plugin .so file.
6. Run the command from above using the custom built qemu-system-aarch64. You need to add a couple of new arguments to enable the vAccel device:
```
-object acceldev-backend-vaccel,id=gen0 \
-device virtio-accel-pci,id=accl0,runtime=gen0,disable-legacy=off,disable-modern=on,event_idx=off \
```

The vAccel backend requires a mat_kernel_size.json file to be able to use it for inference. It expects this to be placed in the `/root/config` directory. You can generate one for the [Llama-3.2-1B-Instruct-f16.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-f16.gguf) model and a given batch size using [scripts/mat_kernel_size_generator.py](scripts/mat_kernel_size_generator.py).


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
