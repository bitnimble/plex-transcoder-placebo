#!/bin/bash
# Builds Plex Transcoder from Plex's published GPL source with Vulkan + libplacebo enabled.
#
# Plex ships ffmpeg 6.1.3, which already contains vf_libplacebo and the whole Vulkan stack;
# they simply omit the flags at build time. So this is mostly a flag flip, plus the output
# drain in patches/ for as long as Plex's base is 6.x and still needs it.
set -euo pipefail

SRC_URL="${SRC_URL:-https://downloads.plex.tv/ffmpeg-source/}"
WORK="${WORK:-$(pwd)/work}"
OUT="${OUT:-$(pwd)/dist}"
PATCH_DIR="${PATCH_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/patches}"

mkdir -p "$WORK" "$OUT"

echo "==> Fetching Plex transcoder source"
curl -sSL --compressed --max-time 600 "$SRC_URL" -o "$WORK/plex-ffmpeg.tar.gz"
gunzip -c "$WORK/plex-ffmpeg.tar.gz" > "$WORK/plex-ffmpeg.tar"

# sed rather than `head -1`: head exits early, tar takes SIGPIPE, pipefail kills the build.
SRCDIR=$(tar -tf "$WORK/plex-ffmpeg.tar" | sed -n '1{s|/.*||;p;}')
# Tarball dir is plexinc-plex-media-server-ffmpeg-gpl-<sha>; the sha prefix also names the
# Codecs/ dir of the matching PMS build, which is how you confirm source matches your server.
PLEX_SHA="${SRCDIR##*-}"
echo "==> Source: $SRCDIR (sha ${PLEX_SHA:0:7})"

# Plex serves only "latest", so what arrives can differ from what the caller resolved
# earlier. Fail loudly rather than publishing an image tagged with a sha it isn't.
EXPECT_SHA="${EXPECT_SHA:-unpinned}"
if [ "$EXPECT_SHA" != unpinned ] && [ "$EXPECT_SHA" != "$PLEX_SHA" ]; then
    echo "ERROR: expected source sha $EXPECT_SHA but got $PLEX_SHA"
    exit 1
fi

rm -rf "${WORK:?}/$SRCDIR"
tar -xf "$WORK/plex-ffmpeg.tar" -C "$WORK"
cd "$WORK/$SRCDIR"

# Sanity: bail loudly if Plex ever drops to a base without libplacebo, rather than
# silently producing a transcoder that can't tone map. Ahead of the patches, so a base
# change reports itself rather than surfacing as a hunk that won't apply.
test -f libavfilter/vf_libplacebo.c || { echo "ERROR: vf_libplacebo.c absent; base changed"; exit 1; }

# Plex serves only "latest", so the base moves under us without warning. The patches fit
# ffmpeg 6.x fftools alone: 7.0 replaced that scheduler with per-filtergraph threads, which
# removes the starvation they work around, so on 7+ the right answer is to skip them and
# keep building rather than to fail every nightly until someone deletes them.
#
# An unreadable version is an error rather than a skip. Guessing wrong in that direction
# ships a 6.x binary without the drain fix, which brings back the 4 GiB subtitle leak with
# nothing anywhere to notice.
test -f RELEASE || { echo "ERROR: no RELEASE file; cannot tell which ffmpeg this is"; exit 1; }
FFMPEG_RELEASE=$(tr -d '[:space:]' < RELEASE)
FFMPEG_MAJOR="${FFMPEG_RELEASE%%.*}"
case $FFMPEG_MAJOR in
    ''|*[!0-9]*) echo "ERROR: unrecognised ffmpeg release '$FFMPEG_RELEASE'"; exit 1 ;;
esac

PATCHED=0
if [ "$FFMPEG_MAJOR" -ge 7 ]; then
    echo "==> ffmpeg $FFMPEG_RELEASE: skipping patches, 7.0+ schedules filtergraphs itself"
elif [ "$FFMPEG_MAJOR" -eq 6 ]; then
    ls "$PATCH_DIR"/*.patch >/dev/null 2>&1 \
      || { echo "ERROR: no patches found in $PATCH_DIR"; exit 1; }
    for patch_file in "$PATCH_DIR"/*.patch; do
        echo "==> Applying ${patch_file##*/}"
        patch --batch --fuzz=0 -p1 < "$patch_file"
    done
    PATCHED=1
else
    echo "ERROR: ffmpeg $FFMPEG_RELEASE predates the tested base"
    exit 1
fi

echo "==> Configuring"
# Deviations from Plex's own configure, and why:
#  - no --enable-gpl: Plex builds LGPL (--enable-openssl would conflict). libplacebo is LGPL.
#  - decoders/encoders left at ffmpeg defaults instead of Plex's hand-picked minimal set.
#    Plex externalises h264/hevc as dlopen'd musl-linked blobs for patent reasons; compiling
#    them in avoids that ABI mess entirely (wrapper must unset FFMPEG_EXTERNAL_LIBS).
#  - no --fatal-warnings: their clang/musl toolchain is clean, gcc/glibc is not.
#  - static (ffmpeg default) rather than their --enable-shared + XORIGIN rpath dance.
# Explicitly bash: Plex's "Fix hwaccel autodetection" block uses ${var/pat/sub}, which dash
# (Ubuntu's /bin/sh) rejects with "Bad substitution".
bash ./configure \
  --prefix="$OUT" \
  --enable-vulkan \
  --enable-libplacebo \
  --enable-libshaderc \
  --enable-vaapi \
  --enable-libdrm \
  --enable-libass \
  --enable-libdav1d \
  --enable-libopus \
  --enable-libvorbis \
  --enable-libxml2 \
  --enable-openssl \
  --enable-eae \
  --disable-doc \
  --disable-ffplay \
  --disable-debug

echo "==> Building"
make -j"$(nproc)"

install -Dm755 ffmpeg "$OUT/Plex Transcoder"

# Write to a file rather than piping: `grep -q` exits on first match, the binary takes
# SIGPIPE, and pipefail reports that as a failed build.
"$OUT/Plex Transcoder" -hide_banner -filters > "$WORK/filters.txt"
grep -q libplacebo "$WORK/filters.txt" \
  || { echo "ERROR: built binary has no libplacebo filter"; exit 1; }

echo "$PLEX_SHA" > "$OUT/PLEX_SOURCE_SHA"

# LGPL: a binary must be accompanied by its corresponding source. Plex's URL only ever
# serves "latest", so linking to it would rot the moment they publish again -- ship the
# exact tarball this was built from instead, and let it travel with the binary. Tarball
# plus patches rather than a re-rolled patched tree: a locally created tar varies run to
# run, so its sha256 would prove nothing, while Plex's own bytes stay checkable.
cp "$WORK/plex-ffmpeg.tar.gz" "$OUT/plex-ffmpeg-source-$PLEX_SHA.tar.gz"
TARBALL_SHA256=$(sha256sum "$WORK/plex-ffmpeg.tar.gz" | cut -d' ' -f1)
install -Dm644 LICENSE.md "$OUT/licenses/ffmpeg-LICENSE.md"
install -Dm644 COPYING.LGPLv2.1 "$OUT/licenses/COPYING.LGPLv2.1"
if [ "$PATCHED" = 1 ]; then
    for patch_file in "$PATCH_DIR"/*.patch; do
        install -Dm644 "$patch_file" "$OUT/licenses/patches/${patch_file##*/}"
    done
    PATCH_NOTE='It also carries the patches in patches/ beside this file. Unpack the tarball
and apply every one with "patch -p1" from the top of the source tree to reproduce it.'
    CORRESPONDING="plex-ffmpeg-source-$PLEX_SHA.tar.gz plus patches/, beside this file."
else
    PATCH_NOTE="No patches were applied: ffmpeg $FFMPEG_RELEASE does not need them."
    CORRESPONDING="plex-ffmpeg-source-$PLEX_SHA.tar.gz, beside this file."
fi
cat > "$OUT/licenses/SOURCE.txt" <<EOF
This binary is a build of Plex's published GPL/LGPL ffmpeg source with the configure flags
in build.sh (notably --enable-vulkan --enable-libplacebo).

$PATCH_NOTE

Upstream release    : $FFMPEG_RELEASE
Upstream source sha : $PLEX_SHA
Obtained from       : $SRC_URL
Tarball sha256      : $TARBALL_SHA256
Corresponding source: $CORRESPONDING

ffmpeg is licensed LGPL v2.1 or later; see COPYING.LGPLv2.1 and ffmpeg-LICENSE.md.
This build does not enable --enable-gpl.
EOF
echo "==> Done: $OUT/Plex Transcoder (source sha ${PLEX_SHA:0:7})"
