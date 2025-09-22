{
    description = "Flake to build simple Linux VMs";
    
    inputs =
    {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
        flake-utils.url = "github:numtide/flake-utils";
	srvos.url = "github:nix-community/srvos";
	srvos.inputs.nixpkgs.follows = "nixpkgs";
    };
    
    outputs = 
    {
        self
        , nixpkgs
        , flake-utils
	, srvos
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
                config = self.nixosConfigurations.linux-image.config;
                inherit (pkgs) lib;
                inherit pkgs;
                partitionTableType = "efi";
                format = "qcow2";
            };
        };
        
	devShells = {
            default = pkgs.mkShell {
                name = "lros-devshell";
                buildInputs = with pkgs;
                [
                    python3
                    gdb
                    qemu_full
                    just
                ];
                LINUX="${pkgs.linuxPackages_latest.kernel}";
            };
        };
    })) // (let
        system = "aarch64-linux";
        pkgs = nixpkgs.legacyPackages.${system};
        kernelPackages = pkgs.linuxPackages_latest; #pkgs.linuxKernel.packages.linux_6_6;
    in{
        nixosConfigurations = {
            linux-image = inputs.nixpkgs.lib.nixosSystem {
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
