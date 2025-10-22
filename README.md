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
