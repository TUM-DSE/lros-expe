# Local VM lifecycle: image build, boot, and SSH access.
# Not standalone: relies on `proot` from the importing justfile.

qemu_ssh_port := "2222"

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

vm nb_cpu="1" size_mem="2G" cpu_type="big" additional_args="":
    #!/usr/bin/env bash
    let "taskset_cores_start = {{ if cpu_type == "big" { "4" } else { "0" } }}"
    let "taskset_cores_end = taskset_cores_start+{{nb_cpu}}-1"
    taskset -c $taskset_cores_start-$taskset_cores_end qemu-kvm \
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
        -nographic {{additional_args}}

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
