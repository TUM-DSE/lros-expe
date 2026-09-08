{
  description = "Flake to build simple Linux VMs";

  inputs =
    {
      nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
      flake-utils.url = "github:numtide/flake-utils";
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
      kernelPackages = pkgs.linuxKernel.packages.linux_6_18;

      vaccel = lros-qemu.packages.${system}.vaccel;
      qemuVaccel = lros-qemu.packages.${system}.qemu-vaccel;
      rtArch = if system == "x86_64-linux" then "x86_64" else "aarch64";

      # The one file and not the whole directory: with ./vaccel_plugins as src,
      # editing a plugin source rebuilt this and moved its store path, which is
      # how the path hardcoded in scripts/exp/targets.sh went stale.
      librknnrt = pkgs.stdenv.mkDerivation {
        pname = "librknnrt";
        version = "unknown";
        src = ./vaccel_plugins/librknnrt.so;
        nativeBuildInputs = [ pkgs.autoPatchelfHook ];
        buildInputs = [ pkgs.stdenv.cc.cc.lib ];
        dontUnpack = true;
        dontBuild = true;
        installPhase = "install -Dm755 $src $out/lib/librknnrt.so";
      };

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

      firmware =
        if system == "aarch64-linux" then {
          AAVMF_CODE = "${pkgs.OVMF.fd}/FV/AAVMF_CODE.fd";
          AAVMF_VARS = "${pkgs.OVMF.fd}/FV/AAVMF_VARS.fd";
        } else {
          AAVMF_CODE = "${pkgs.qemu}/share/qemu/edk2-aarch64-code.fd";
          AAVMF_VARS = pkgs.runCommand "aavmf-vars.fd" { } "install -m444 /dev/null $out";
        };

      pluginEnv = {
        VACCEL_PLUGINS_RKNN = lib.optionalString (system == "aarch64-linux")
          "${vaccel-plugin-rknn}/lib/libvaccel-rknn.so";
        VACCEL_PLUGINS_CUDA = "";
      };

      # CUDA is unfree, so it needs its own import rather than legacyPackages.
      # 12.6 and not the default: the Orin's driver is the JetPack one, and a
      # binary built against a newer toolkit fails at load.
      cuda = (import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      }).cudaPackages_12_6;

      # Everything vaccel_plugins/build-plugin.sh reads, so that it carries no
      # store paths of its own and the two boards get theirs from this lock.
      # getDev/getLib rather than .dev/.lib: the CUDA packages have split those
      # outputs in some nixpkgs revisions and not others.
      pluginShellEnv = {
        VACCEL_PREFIX = "${vaccel}";
      };
      cudaShellEnv = {
        CUDART = "${lib.getLib cuda.cuda_cudart}/lib";
        CUBLAS = "${lib.getLib cuda.libcublas}/lib";
        CUDA_DEV_INCLUDE = lib.concatStringsSep " " [
          "-I${lib.getDev cuda.cuda_cudart}/include"
          "-I${lib.getDev cuda.libcublas}/include"
          "-I${lib.getDev cuda.cuda_cccl}/include"
        ];
        CUDA_ARCH = "87";
      };
    in {
      devShells = {
        default = pkgs.mkShell.override { stdenv = pkgs.gcc13Stdenv; } ({
          name = "lros-devshell";
          buildInputs = with pkgs;
          [
            qemuVaccel
            vaccel
            cmake
            ninja
            pkg-config
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
            e2fsprogs
          ];
          QEMU_VACCEL = "${qemuVaccel}/bin/qemu-system-${rtArch}";
          # not the plain nixpkgs one.
          LINUX = if system == "aarch64-linux"
            then "${self.nixosConfigurations.linux-conf.config.boot.kernelPackages.kernel}"
            else "${kernelPackages.kernel}";
          LIBSTOP="${libstop}/lib/libstop.so";
          shellHook = lib.optionalString (system == "aarch64-linux") ''
            export CONF=$(nix eval --raw .#nixosConfigurations.linux-conf.config.system.build.toplevel)
          '';
        } // pluginEnv // firmware);
      } // lib.optionalAttrs (system == "aarch64-linux") {
        # The environment vaccel_plugins/build.sh runs build-plugin.sh in, one
        # per board, so neither script carries a store path. Separate shells
        # because they share a system: pulling CUDA into the RK3588's shell
        # would download a toolkit it has no use for.
        plugin-rk3588 = pkgs.mkShell ({
          name = "lros-plugin-rk3588";
          buildInputs = [ vaccel librknnrt pkgs.cmake pkgs.ninja ];
          PLATFORM = "rk3588";
          RKNN = "${librknnrt}";
        } // pluginShellEnv);

        # gcc13, because nvcc 12.6 rejects anything newer with a host_config.h
        # version check, and that is what broke the link when the ggml stage's
        # toolchain reached it.
        plugin-orin = pkgs.mkShell.override { stdenv = pkgs.gcc13Stdenv; } ({
          name = "lros-plugin-orin";
          buildInputs = [ vaccel pkgs.cmake pkgs.ninja cuda.cuda_nvcc cuda.cudatoolkit ];
          PLATFORM = "orin";
          CUDAToolkit_ROOT = "${cuda.cudatoolkit}";
          # nvcc resolves its own symlink and looks for headers beside the real
          # binary, which in the joined toolkit has no cuda_runtime.h.
          NVCC_PREPEND_FLAGS = "-I${cuda.cudatoolkit}/include";
        } // pluginShellEnv // cudaShellEnv);
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
