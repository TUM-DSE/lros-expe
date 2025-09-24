# This configuration describes a template Linux VM image
{ lib, pkgs, kernelPackages, ... }:
with lib;

let # this executable enables to fit the VM shell to your terminal
  resize = pkgs.writeShellScriptBin "resize" ''
    export PATH=${pkgs.coreutils}/bin
    if [ ! -t 0 ]; then
      # not a interactive...
      exit 0
    fi
    TTY="$(tty)"
    if [[ "$TTY" != /dev/ttyS* ]] && [[ "$TTY" != /dev/ttyAMA* ]] && [[ "$TTY" != /dev/ttySIF* ]]; then
      # probably not a known serial console, we could make this check more
      # precise by using `setserial` but this would require some additional
      # dependency
      exit 0
    fi
    old=$(stty -g)
    stty raw -echo min 0 time 5

    printf '\0337\033[r\033[999;999H\033[6n\0338' > /dev/tty
    IFS='[;R' read -r _ rows cols _ < /dev/tty

    stty "$old"
    stty cols "$cols" rows "$rows"
  '';
in
{
  networking = {
    hostName = "guest";
  };
  services.sshd.enable = true;
  networking.firewall.enable = true;

  users.users.root.password = "password";
  services.openssh.settings.PermitRootLogin = lib.mkDefault "yes";
  users.users.root.openssh.authorizedKeys.keys = [
    (builtins.readFile ./keyfile.pub)
  ];
  services.getty.autologinUser = lib.mkDefault "root";

  fileSystems."/root" = { # this will mount the main directory onto the /root of the guest
      device = "home";
      fsType = "9p";
      options = [ "trans=virtio" "nofail" "msize=104857600" ];
  };

  # mount host nix store, but use overlay fs to make it writeable
  fileSystems."/nix/.ro-store-vmux" = {
    device = "nixstore";
    fsType = "9p";
    options = [ "ro" "trans=virtio" "nofail" "msize=104857600" ];
    neededForBoot = true;
  };
  fileSystems."/nix/store" = {
    device = "overlay";
    fsType = "overlay";
    options = [
      "lowerdir=/nix/.ro-store-vmux"
      "upperdir=/nix/.rw-store/store"
      "workdir=/nix/.rw-store/work"
    ];
    neededForBoot = true;
    depends = [
      "/nix/.ro-store-vmux"
      "/nix/.rw-store/store"
      "/nix/.rw-store/work"
    ];
  };
  boot.initrd.availableKernelModules = [ "overlay" ];
  boot.kernelParams = [ "console=ttyAMA0,115200" ];

  nix.extraOptions = ''
      experimental-features = nix-command flakes
  '';
  nix.package = pkgs.nixFlakes;
  environment.systemPackages = [
    pkgs.vim
    pkgs.git
    pkgs.gnumake
    pkgs.just
    pkgs.python3
    resize
    kernelPackages.perf
    pkgs.cmake
    pkgs.gcc
    pkgs.llama-cpp
    pkgs.flamegraph
    # Add more packages here if needed
  ];

  # define the Linux version in the main flake.nix
  boot.kernelPackages = kernelPackages;

  system.stateVersion = "24.05";

  # things to improve QoL with the guest shell
  console.enable = true;
  environment.loginShellInit = "${resize}/bin/resize";
  systemd.services."serial-getty@ttyAMA0".enable = true;
  systemd.services."serial-getty@ttyAMA0".environment.TERM = "xterm-256color";
  services.qemuGuest.enable = true;
}
