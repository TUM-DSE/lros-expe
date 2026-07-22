# Dependency/source builds (vAccel, QEMU, llama.cpp) and cleanup.
# Not standalone: relies on `proot` from the importing justfile.

clean_builds:
    #!/usr/bin/env bash
    rm -rf vaccel/build
    rm -rf lros-qemu/build

build_vaccel:
    #!/usr/bin/env bash
    if [ ! -d vaccel/build ]; then
        cd vaccel/scripts/common; git apply ../../submodules.patch 2> /dev/null; cd ../..
        meson setup --buildtype=release build
        meson compile -C build
        meson install -C build --destdir=out
        sed -i "s/prefix=/prefix=$(echo {{proot}} | sed 's/\//\\\//g')\/vaccel\/build\/out/" {{proot}}/vaccel/build/out/usr/local/lib/pkgconfig/vaccel.pc
        echo "Finished building vAccel"
    else
        echo "vAccel is already built"
    fi

build_qemu:
    #!/usr/bin/env bash
    if [ ! -d lros-qemu/build ]; then
        cd lros-qemu; mkdir build; cd build
        CFLAGS=-Wno-error PKG_CONFIG_PATH={{proot}}/vaccel/build/out/usr/local/lib/pkgconfig ../configure --target-list=aarch64-softmmu --enable-virtfs
        make -j
        echo "Finished building Qemu"
    else
        echo "Qemu is already built"
    fi

build_llamacpp:
    #!/usr/bin/env bash
    if [ ! -d lros/llama.cpp-rknn/build ]; then
        cd lros/llama.cpp-rknn
        cmake -B build -DGGML_RKNN=ON
        cmake --build build --config Release --target llama-batched-bench llama-server -j
    else
        echo "llama.cpp is already built"
    fi
