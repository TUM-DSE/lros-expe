{ self
, pkgs
, lib
, selfpkgs
, ...
}:

pkgs.stdenv.mkDerivation {
  pname = "vaccel";
  version = "v" + "0.7.1";
  src = fetchFromGitHub {
    owner = "TUM-DSE";
    repo = "vaccel";
    rev = "cc9942e6f5de46ff1f5eb139a208de5998024d64";
    hash = "";
  };

  cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Release" ];
  postPatch = ''
    substituteInPlace CMakeLists.txt \
      --replace 'set(CMAKE_INSTALL_PREFIX ''${CMAKE_BINARY_DIR}/install)' ""
  '';
  nativeBuildInputs = [ pkgs.cmake pkgs.gnumake ];
  patches = [ ./cstdint.patch ];

  meta = with lib; {
    homepage = https://github.com/LLNL/umap;
    description = "User-space Page Management";
    license = licenses.lgpl2;
    broken = false;
  };
}
