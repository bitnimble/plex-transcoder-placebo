# linuxserver.io Docker Mod: Vulkan/libplacebo HDR tone mapping for Plex on AMD.
#
# Builds Plex's own GPL source with --enable-vulkan --enable-libplacebo, then bundles the
# binary with its libplacebo/Vulkan/RADV stack. Bundling is what makes the mod portable:
# the container's Ubuntu version no longer has to match the build.
#
# 24.04 rather than the container's 26.04 on purpose -- glibc is forward-compatible, so an
# older build runs on newer containers but not the reverse.
FROM --platform=linux/amd64 ubuntu:24.04 AS build

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates pkg-config nasm patch \
        libplacebo-dev libvulkan-dev libshaderc-dev libva-dev libdrm-dev \
        ocl-icd-opencl-dev libass-dev libdav1d-dev libopus-dev \
        libvorbis-dev libxml2-dev libssl-dev libzvbi-dev \
        mesa-vulkan-drivers mesa-va-drivers libdrm-common \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY build.sh collect-libs.sh ./
COPY patches/ ./patches/

# build.sh fetches whatever Plex is currently serving, but Docker caches this layer on the
# script's contents alone. Without the sha in the layer key, a rebuild triggered by a new
# Plex release would hit the cache and produce the OLD binary -- silently defeating the
# whole point of the scheduled rebuild. The workflow passes the sha it resolved.
ARG PLEX_SHA=unpinned
RUN EXPECT_SHA="$PLEX_SHA" bash build.sh
RUN bash collect-libs.sh "/src/dist/Plex Transcoder" /source/plex-placebo \
    && install -Dm755 "/src/dist/Plex Transcoder" /source/plex-placebo/bin/Plex-Transcoder-placebo \
    && cp /src/dist/PLEX_SOURCE_SHA /source/plex-placebo/ \
    && cp -r /src/dist/licenses /source/plex-placebo/ \
    && cp /src/dist/plex-ffmpeg-source-*.tar.gz /source/plex-placebo/licenses/

COPY plex-transcoder-wrapper.sh /source/plex-placebo/bin/wrapper.sh
COPY root/ /source/

# libdrm hardcodes this path -- there is no env override -- so amdgpu.ids has to land at
# the real location or every transcode logs "No such file or directory". The vaapi-amdgpu
# mod ships it here for the same reason.
RUN install -Dm644 /usr/share/libdrm/amdgpu.ids /source/usr/share/libdrm/amdgpu.ids \
    && chmod +x /source/plex-placebo/bin/wrapper.sh \
    && find /source/etc/s6-overlay -name run -exec chmod +x {} +

# The lsio mod loader reads `.layers[0].digest` and downloads ONLY the first layer, so the
# final stage must produce exactly one. Adding a second COPY here would silently ship half
# the mod.
FROM scratch
COPY --from=build /source/ /
