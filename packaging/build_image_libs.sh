#!/usr/bin/env bash
# Build the native image libraries torchvision links against (libpng, libwebp,
# libjpeg-turbo, plus zlib on Windows) from source, into one relocatable prefix.
#
# This replaces `conda install libpng libwebp libjpeg-turbo` in
# packaging/pre_build_script.sh. Use the result with:
#   export TORCHVISION_INCLUDE="${PREFIX}/include"
#   export TORCHVISION_LIBRARY="${PREFIX}/lib"
#   export PATH="${PREFIX}/bin:${PATH}"   # setup.py looks for libpng-config / pngfix
#
# Usage: PREFIX=/path/to/prefix packaging/build_image_libs.sh
# Needs: git, cmake >= 3.16, ninja, a C compiler (MSVC on Windows), and nasm on
# x86 (libjpeg-turbo is built with REQUIRE_SIMD, so a missing nasm fails the build
# instead of silently producing a slow library).

set -euxo pipefail

# Pinned releases. Each tag is checked against its commit SHA after cloning.
ZLIB_TAG="v1.3.1"
ZLIB_SHA="51b7f2abdade71cd9bb0e7a373ef2610ec6f9daf"
LIBPNG_TAG="v1.6.59"
LIBPNG_SHA="cd952f49f95bb27154ae77dbb103032d95f6e580"
# torchvision requires libwebp >= 1.3.2 (CVE-2023-4863)
LIBWEBP_TAG="v1.6.0"
LIBWEBP_SHA="4fa21912338357f89e4fd51cf2368325b59e9bd9"
LIBJPEG_TURBO_TAG="3.1.4.1"
LIBJPEG_TURBO_SHA="9217719d3a58633923b096af4c1d50d304768a64"

: "${PREFIX:?PREFIX must be set to the install directory}"
mkdir -p "${PREFIX}"
WORK_DIR="${WORK_DIR:-$(pwd)/image_libs_build}"
mkdir -p "${WORK_DIR}"

case "$(uname -s)" in
  Linux*) OS=linux ;;
  Darwin*) OS=macos ;;
  MINGW* | MSYS* | CYGWIN*) OS=windows ;;
  *) echo "Unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac

if [[ "${OS}" == windows ]]; then
  # CMake is a native Windows binary: give it C:/... paths, not /c/...
  PREFIX="$(cygpath -m "${PREFIX}")"
  WORK_DIR="$(cygpath -m "${WORK_DIR}")"
fi

clone_pinned() {
  local url="$1" tag="$2" sha="$3" dest="$4"
  rm -rf "${dest}"
  git clone --quiet --depth 1 --branch "${tag}" "${url}" "${dest}"
  local actual
  actual="$(git -C "${dest}" rev-parse HEAD)"
  if [[ "${actual}" != "${sha}" ]]; then
    echo "${url} ${tag} is ${actual}, expected ${sha}" >&2
    exit 1
  fi
}

COMMON_ARGS=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="${PREFIX}"
  -DCMAKE_PREFIX_PATH="${PREFIX}"
  -DCMAKE_INSTALL_LIBDIR=lib
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DBUILD_SHARED_LIBS=ON
)
if [[ "${OS}" == linux ]]; then
  # Libraries in the prefix find each other (libwebp -> libsharpyuv); auditwheel
  # relocates them into the wheel later.
  COMMON_ARGS+=("-DCMAKE_INSTALL_RPATH=\$ORIGIN")
elif [[ "${OS}" == macos ]]; then
  COMMON_ARGS+=(
    -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
    -DCMAKE_OSX_ARCHITECTURES="$(uname -m)"
    -DCMAKE_MACOSX_RPATH=ON
    -DCMAKE_INSTALL_RPATH=@loader_path
  )
fi

build_and_install() {
  local name="$1"
  shift
  cmake -S "${WORK_DIR}/src/${name}" -B "${WORK_DIR}/build/${name}" "${COMMON_ARGS[@]}" "$@"
  cmake --build "${WORK_DIR}/build/${name}" --parallel
  cmake --install "${WORK_DIR}/build/${name}"
}

# zlib is a libpng dependency. Linux and macOS use the system zlib; Windows has none.
if [[ "${OS}" == windows ]]; then
  clone_pinned https://github.com/madler/zlib "${ZLIB_TAG}" "${ZLIB_SHA}" "${WORK_DIR}/src/zlib"
  build_and_install zlib
  # Keep the import library only, so libpng cannot pick up the static one
  rm -f "${PREFIX}/lib/zlibstatic.lib"
fi

clone_pinned https://github.com/pnggroup/libpng "${LIBPNG_TAG}" "${LIBPNG_SHA}" "${WORK_DIR}/src/libpng"
build_and_install libpng \
  -DPNG_SHARED=ON \
  -DPNG_STATIC=OFF \
  -DPNG_TESTS=OFF \
  -DPNG_FRAMEWORK=OFF \
  -DPNG_TOOLS=ON  # setup.py finds libpng through pngfix on Windows

clone_pinned https://github.com/webmproject/libwebp "${LIBWEBP_TAG}" "${LIBWEBP_SHA}" "${WORK_DIR}/src/libwebp"
# Library only: the command line tools pull in optional system libraries
# (giflib, libtiff, ...) that would leak into the build.
build_and_install libwebp \
  -DWEBP_BUILD_ANIM_UTILS=OFF \
  -DWEBP_BUILD_CWEBP=OFF \
  -DWEBP_BUILD_DWEBP=OFF \
  -DWEBP_BUILD_GIF2WEBP=OFF \
  -DWEBP_BUILD_IMG2WEBP=OFF \
  -DWEBP_BUILD_VWEBP=OFF \
  -DWEBP_BUILD_WEBPINFO=OFF \
  -DWEBP_BUILD_WEBPMUX=OFF \
  -DWEBP_BUILD_EXTRAS=OFF

clone_pinned https://github.com/libjpeg-turbo/libjpeg-turbo "${LIBJPEG_TURBO_TAG}" "${LIBJPEG_TURBO_SHA}" \
  "${WORK_DIR}/src/libjpeg-turbo"
build_and_install libjpeg-turbo \
  -DENABLE_SHARED=ON \
  -DENABLE_STATIC=OFF \
  -DWITH_JPEG8=1 \
  -DWITH_TURBOJPEG=OFF \
  -DREQUIRE_SIMD=1

if [[ "${OS}" == windows ]]; then
  # setup.py links "libpng" and "libwebp" on Windows. The aliases are import
  # libraries only; the DLLs keep their original names.
  [[ -f "${PREFIX}/lib/libpng.lib" ]] || cp "${PREFIX}/lib/libpng16.lib" "${PREFIX}/lib/libpng.lib"
  [[ -f "${PREFIX}/lib/libwebp.lib" ]] || cp "${PREFIX}/lib/webp.lib" "${PREFIX}/lib/libwebp.lib"
fi

# Check everything setup.py will look for is in the prefix.
require() {
  if ! compgen -G "$1" > /dev/null; then
    echo "Missing from prefix: $1" >&2
    exit 1
  fi
}
require "${PREFIX}/include/png.h"
require "${PREFIX}/include/jpeglib.h"
require "${PREFIX}/include/webp/decode.h"
if [[ "${OS}" == windows ]]; then
  require "${PREFIX}/bin/pngfix.exe"
  require "${PREFIX}/lib/libpng.lib"
  require "${PREFIX}/lib/libwebp.lib"
  require "${PREFIX}/lib/jpeg.lib"
  require "${PREFIX}/bin/*.dll"
else
  require "${PREFIX}/bin/libpng-config"
  case "${OS}" in
    linux) ext=so ;;
    macos) ext=dylib ;;
  esac
  require "${PREFIX}/lib/libpng*.${ext}*"
  require "${PREFIX}/lib/libwebp*.${ext}*"
  require "${PREFIX}/lib/libjpeg*.${ext}*"

  # Compile, link and run against the prefix with the same flags setup.py uses.
  cat > "${WORK_DIR}/check.c" <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <jpeglib.h>
#include <webp/decode.h>
int main(void) {
  int webp = WebPGetDecoderVersion();
  printf("libpng %s, libjpeg API %d, libwebp 0x%06x\n",
         PNG_LIBPNG_VER_STRING, JPEG_LIB_VERSION, webp);
  return webp >= 0x010302 ? 0 : 1;  // libwebp >= 1.3.2
}
EOF
  "${CC:-cc}" "${WORK_DIR}/check.c" -o "${WORK_DIR}/check" \
    -I"${PREFIX}/include" -L"${PREFIX}/lib" -lpng -ljpeg -lwebp -Wl,-rpath,"${PREFIX}/lib"
  "${WORK_DIR}/check"
  "${PREFIX}/bin/libpng-config" --version
fi

# Record what is in the bundle.
{
  echo "libpng ${LIBPNG_TAG} ${LIBPNG_SHA}"
  echo "libwebp ${LIBWEBP_TAG} ${LIBWEBP_SHA}"
  echo "libjpeg-turbo ${LIBJPEG_TURBO_TAG} ${LIBJPEG_TURBO_SHA}"
  [[ "${OS}" != windows ]] || echo "zlib ${ZLIB_TAG} ${ZLIB_SHA}"
  echo "built on $(uname -srm)"
  echo
  (cd "${PREFIX}" && find lib bin -type f -o -type l | sort)
} > "${PREFIX}/MANIFEST.txt"
cat "${PREFIX}/MANIFEST.txt"
