proot := justfile_directory()
qemu_ssh_port := "2222"

models_dir := proot+"/models"
models_to_get := "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-f16.gguf"

default:
    @just --choose

help:
    just --list

ssh_local cmd="":
    ssh \
    -i {{proot}}/nix/keyfile \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o IdentityAgent=/dev/null \
    -F /dev/null \
    -p {{qemu_ssh_port}} \
    root@localhost -- "{{cmd}}"

ssh_remote host="" cmd="":
    ssh \
    -i {{proot}}/nix/keyfile \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o IdentityAgent=/dev/null \
    -F /dev/null \
    {{host}} -- "{{cmd}}"

vm nb_cpu="1" size_mem="2G" cpu_type="big":
    #!/usr/bin/env bash
    let "taskset_cores_start = {{ if cpu_type == "big" { 4 } else { 0 } }}"
    let "taskset_cores_end = taskset_cores_start+{{nb_cpu}}-1"
    taskset -c 0-$taskset_cores qemu-kvm \
        -cpu host \
        -smp {{nb_cpu}} \
        -m {{size_mem}} \
        -machine virt \
        -kernel $LINUX/Image \
        -append "root=/dev/vda2 init="$CONF"/init console=ttyAMA0,115200" \
        -device virtio-serial \
        -fsdev local,id=home,path={{proot}},security_model=none \
        -device virtio-9p-pci,fsdev=home,mount_tag=home,disable-modern=on,disable-legacy=off \
        -fsdev local,id=nixstore,path=/nix/store,security_model=none \
        -device virtio-9p-pci,fsdev=nixstore,mount_tag=nixstore,disable-modern=on,disable-legacy=off \
        -drive file={{proot}}/VMs/linux-image.qcow2 \
        -net nic,netdev=user.0,model=virtio \
        -netdev user,id=user.0,hostfwd=tcp:127.0.0.1:{{qemu_ssh_port}}-:22 \
        -nographic

vm-image-init:
    #!/usr/bin/env bash
    set -x
    set -e
    echo "Initializing disk for the VM"
    mkdir -p {{proot}}/VMs

    # build images fast
    overwrite() {
        install -D -m644 {{proot}}/VMs/ro/nixos.qcow2 {{proot}}/VMs/$1.qcow2
        qemu-img resize {{proot}}/VMs/$1.qcow2 +8g
    }

    taskset -c 0-3 nix build .#linux-image --out-link {{proot}}/VMs/ro
    overwrite linux-image
    export CONF=$(nix eval --raw .#nixosConfigurations.linux-conf.config.system.build.toplevel)

get_models:
	#!/usr/bin/env bash
	mkdir -p {{models_dir}}
	declare -A models={{models_to_get}}
	for url in $models; do
		name=$(echo $url | awk -F '/' '{print $(NF)})'
		echo $name
		#if [ -f "$models_dir/$name" ]; then
		#    echo "$name already exists, skipping."
		#else
		#    wget -q --show-progress -O "{{models_dir}}/$name" "$url"
		#fi
	done

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

