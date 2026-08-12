{
  description = "Flake to build simple Linux VMs";

  inputs =
    {
      nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
      flake-utils.url = "github:numtide/flake-utils";
      # QEMU with the virtio-accel device, plus the host-side vAccel runtime it
      # links against. The branch matters: master+vaccel+modern is the one where
      # disable-legacy/disable-modern pick the transport instead of the device
      # forcing legacy for Unikraft's benefit. miniOSv needs virtio-1 on
      # aarch64, where a legacy device's I/O BAR is unreachable.
      lros-qemu.url = "github:TUM-DSE/lros-qemu/master+vaccel+modern";
    };

    outputs =
    {
      self
      , nixpkgs
      , flake-utils
      , lros-qemu
    } @ inputs:
    (flake-utils.lib.eachSystem ["x86_64-linux" "aarch64-linux"] (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      inherit (pkgs) lib;
      make-disk-image = import (./nix/make-disk-image.nix);
      # Same kernel as the VM image below.
      kernelPackages = pkgs.linuxKernel.packages.linux_6_18;

      vaccel = lros-qemu.packages.${system}.vaccel;
      qemuVaccel = lros-qemu.packages.${system}.qemu-vaccel;
      rtArch = if system == "x86_64-linux" then "x86_64" else "aarch64";

      # The RK3588 NPU's userspace driver: a vendor blob checked into
      # vaccel_plugins/. autoPatchelfHook points its DT_NEEDEDs at this nixpkgs'
      # libc/libstdc++ -- without that it resolves nothing inside a nix-built
      # QEMU, which is its own loader with its own search path.
      librknnrt = pkgs.stdenv.mkDerivation {
        pname = "librknnrt";
        version = "unknown";
        src = ./vaccel_plugins;
        nativeBuildInputs = [ pkgs.autoPatchelfHook ];
        buildInputs = [ pkgs.stdenv.cc.cc.lib ];
        dontBuild = true;
        installPhase = "install -Dm755 librknnrt.so $out/lib/librknnrt.so";
      };

      # The vAccel plugin QEMU's acceldev backend dlopens: it turns the guest's
      # matmul ops into RKNN calls. Built with $CC rather than through
      # vaccel_plugins/CMakeLists.txt, which declares `LANGUAGES C CXX CUDA` and
      # a required CUDAToolkit and so cannot configure on a board that has an
      # NPU and no GPU.
      vaccel-plugin-rknn = pkgs.stdenv.mkDerivation {
        pname = "vaccel-plugin-rknn";
        version = "0.7.1";
        src = ./vaccel_plugins;
        buildInputs = [ vaccel librknnrt ];
        buildPhase = ''
          $CC -shared -fPIC -O2 -o libvaccel-rknn.so rknn.c -I. -lvaccel -lrknnrt
        '';
        installPhase = "install -Dm755 libvaccel-rknn.so $out/lib/libvaccel-rknn.so";
        meta.platforms = [ "aarch64-linux" ];
      };

      libstop = pkgs.stdenv.mkDerivation {
        name = "libstop";
        src = ./benchmarks/boottime;
        buildPhase = ''
          $CC -shared -fPIC -O2 stop.c -o libstop.so
        '';
        installPhase = ''
          install -Dm755 libstop.so $out/lib/libstop.so
        '';
      };

      # UEFI firmware for the unikernel, which boots as an EFI application:
      # miniosv/scripts/run.py looks for AAVMF_CODE/AAVMF_VARS. An x86_64 host
      # has no AAVMF package, but nixpkgs' qemu ships an aarch64 image, which is
      # enough to boot the aarch64 guest under TCG for a smoke test.
      firmware =
        if system == "aarch64-linux" then {
          AAVMF_CODE = "${pkgs.OVMF.fd}/FV/AAVMF_CODE.fd";
          AAVMF_VARS = "${pkgs.OVMF.fd}/FV/AAVMF_VARS.fd";
        } else {
          AAVMF_CODE = "${pkgs.qemu}/share/qemu/edk2-aarch64-code.fd";
          AAVMF_VARS = pkgs.runCommand "aavmf-vars.fd" { } "install -m444 /dev/null $out";
        };

      # Consumed by bench.just, which picks between them by probing for the NPU
      # or the NVIDIA driver. Empty where the accelerator cannot be there.
      # VACCEL_PLUGINS_CUDA is not wired up yet: the CUDA plugin (cuda.cu) has
      # not been rebuilt against this vAccel, and the Jetson is offline.
      pluginEnv = {
        VACCEL_PLUGINS_RKNN = lib.optionalString (system == "aarch64-linux")
          "${vaccel-plugin-rknn}/lib/libvaccel-rknn.so";
        VACCEL_PLUGINS_CUDA = "";
      };
    in {
      devShells = {
        default = pkgs.mkShell.override { stdenv = pkgs.gcc13Stdenv; } ({
          name = "lros-devshell";
          buildInputs = with pkgs;
          [
            # qemu-vaccel provides qemu-system-<arch> and the qemu-kvm symlink
            # vm.just calls, so it is the only QEMU on PATH -- the unikernel and
            # the Linux VM are then measured under the same binary.
            qemuVaccel
            vaccel
            # The native baselines are built from llama.cpp-rknn with its own
            # CMakePresets.json, which declares "version": 4 and so needs
            # cmake >= 3.23. This used to arrive transitively through the lros
            # flake's dev shell; with that gone it has to be named here, or the
            # host's cmake gets used and fails with "Unrecognized version field".
            cmake
            ninja
            pkg-config
            # The shellHook below shells out to `nix`. On the Orange Pi the
            # system one is 2.6.0, which cannot even parse a flake ref with a
            # '+' in the branch name. Ship a current one rather than depend on
            # whatever the board happens to have.
            nix
            ack
            python3
            gdb
            just
            python312Packages.tomli
            python312Packages.pyusb
            python312Packages.pandas
            python312Packages.matplotlib
            python312Packages.seaborn
            python312Packages.crc
            bc
            stress
            ncurses
            vmtouch
            bpftrace
            # Building the unikernel data disk (models + config) as ext4.
            e2fsprogs
          ];
          # miniosv/scripts/run.py reaches for these by name.
          QEMU_VACCEL = "${qemuVaccel}/bin/qemu-system-${rtArch}";
          # The vm recipe boots this directly via -kernel: use the image's
          # kernel, which includes the boot-event patches (trace_hvc etc.),
          # not the plain nixpkgs one.
          LINUX = if system == "aarch64-linux"
            then "${self.nixosConfigurations.linux-conf.config.boot.kernelPackages.kernel}"
            else "${kernelPackages.kernel}";
          LIBSTOP="${libstop}/lib/libstop.so";
          shellHook = lib.optionalString (system == "aarch64-linux") ''
            export CONF=$(nix eval --raw .#nixosConfigurations.linux-conf.config.system.build.toplevel)
          '';
        } // pluginEnv // firmware);
      };
    } // lib.optionalAttrs (system == "aarch64-linux") {
      packages =
      {
        inherit vaccel librknnrt vaccel-plugin-rknn;
        qemu-vaccel = qemuVaccel;

        linux-image = make-disk-image {
          config = self.nixosConfigurations.linux-conf.config;
          inherit (pkgs) lib;
          inherit pkgs;
          partitionTableType = "efi";
          format = "qcow2";
        };
      };
    })) // (let
      system = "aarch64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      kernelPackages = pkgs.linuxKernel.packages.linux_6_18;
    in{
      nixosConfigurations = {
        linux-conf = inputs.nixpkgs.lib.nixosSystem {
          inherit system;
          modules = [
            (import ./nix/image.nix
            {
              inherit pkgs;
              inherit (pkgs) lib;
              inherit kernelPackages;
            })
            ./nix/nixos-generators-qcow.nix
          ];
        };
      };
    });
  }
