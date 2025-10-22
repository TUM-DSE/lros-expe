{ self
, pkgs
, lib
, selfpkgs
, ...
}:

pkgs.stdenv.mkDerivation {
  pname = "vaccel";
  version = "v" + "0.7.1";
  src = pkgs.fetchFromGitHub {
    owner = "TUM-DSE";
    repo = "vaccel";
    rev = "cc9942e6f5de46ff1f5eb139a208de5998024d64";
    hash = "sha256-3CZfSGQtCcFlZZiVHiZSoFbHbKOg/GtVVKlSWWeCq0k=";
    fetchSubmodules = true;
  };

  nativeBuildInputs = [ pkgs.meson pkgs.ninja pkgs.git pkgs.pkg-config ];
  
  patchPhase = ''
    cd scripts/common
    git apply ../../submodules.patch
  '';

  mesonFlags = [
    "--buildtype=release"
  ];
  mesonBuildDir = "nix-build";

  installPhase = ''
    mkdir -p $out
    cp tst $out
  '';

  meta = with lib; {
    homepage = "https://vaccel.org/";
    description = "vAccel is a runtime library that aims to help development of applications using hardware acceleration";
    license = licenses.lgpl2;
  };
}
