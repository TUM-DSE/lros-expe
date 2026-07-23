{
  description = "Flake to build simple Linux VMs";

  inputs =
    {
      nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
      flake-utils.url = "github:numtide/flake-utils";
      # The lros submodule's self-contained flake (patched qemu, vaccel,
      # plugins). git+file (not path:lros/flake) copies only lros's own tree, so
      # pure eval works. It's a locked input: re-lock after editing lros's flake.
      # On Nix >= 2.35 this can become `self.submodules = true` + path:lros/flake.
      lros.url = "git+file:./lros?dir=flake";
    };

    outputs =
    {
      self
      , nixpkgs
      , flake-utils
      , lros
    } @ inputs:
    (flake-utils.lib.eachSystem ["x86_64-linux" "aarch64-linux"] (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      inherit (pkgs) lib;
      make-disk-image = import (./nix/make-disk-image.nix);
      selfpkgs = self.packages.${system};
      kernelPackages = pkgs.linuxKernel.packages.linux_6_6;
    in {
      devShells = {
        default = pkgs.mkShell.override { stdenv = pkgs.gcc13Stdenv; } ({
          name = "lros-devshell";
          # Pull in the lros flake's toolchain (patched qemu, vaccel, plugins, …).
          inputsFrom = [ lros.devShells.${system}.default ];
          buildInputs = with pkgs;
          [
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
          ];
          LINUX="${pkgs.linuxPackages_latest.kernel}";
          shellHook = lib.optionalString (system == "aarch64-linux") ''
            export CONF=$(nix eval --raw .#nixosConfigurations.linux-conf.config.system.build.toplevel)
          '';
        # VACCEL_PLUGINS_* from the lros flake (empty on x86_64).
        } // lros.pluginEnv.${system});
      };
    } // lib.optionalAttrs (system == "aarch64-linux") {
      packages =
      {
        linux-image = make-disk-image {
          config = self.nixosConfigurations.linux-conf.config;
          inherit (pkgs) lib;
          inherit pkgs;
          partitionTableType = "efi";
          format = "qcow2";
        };

        vaccel = pkgs.callPackage ./nix/vaccel.nix {
          inherit pkgs;
          inherit inputs;
          inherit selfpkgs;
          inherit self;
        };
      };
    })) // (let
      system = "aarch64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      kernelPackages = pkgs.linuxPackages_latest; #pkgs.linuxKernel.packages.linux_6_6;
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
