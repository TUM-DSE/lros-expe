{
  description = "Flake to build simple Linux VMs";

  inputs =
    {
      nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
      flake-utils.url = "github:numtide/flake-utils";
    };

    outputs = 
    {
      self
      , nixpkgs
      , flake-utils
    } @ inputs:
    (flake-utils.lib.eachSystem ["aarch64-linux"] (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      make-disk-image = import (./nix/make-disk-image.nix);
      selfpkgs = self.packages.${system};
      kernelPackages = pkgs.linuxKernel.packages.linux_6_6;
    in {
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

      devShells = {
        default = pkgs.mkShell {
          name = "lros-devshell";
          buildInputs = with pkgs;
          [
            ack
            python3
            gdb
            qemu_full
            just
            meson
            ninja
            glib.dev
            pkg-config
            python312Packages.tomli
            python312Packages.pyusb
            python312Packages.crc
            bc
            stress
          ];
          LINUX="${pkgs.linuxPackages_latest.kernel}";
          shellHook = ''
            export CONF=$(nix eval --raw .#nixosConfigurations.linux-conf.config.system.build.toplevel)
          '';
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
