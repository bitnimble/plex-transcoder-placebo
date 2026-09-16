#!/bin/sh
# Swaps Plex's software tone mapping for GPU libplacebo, and chains to whatever binary was
# at this path before us for everything else.
#
# -f (noglob): filter graphs are full of [label] which are glob patterns. Nothing here wants
# pathname expansion, and the graph-reassembly loop splits an unquoted string, so a file in
# Plex's CWD matching a whole segment would otherwise corrupt the graph on the no-fallback
# exec path. Disable it for the whole script rather than fencing one loop.
set -uf

# ${0%/*} rather than $(dirname "$0"): a command substitution here runs before every path
# through this script, so a broken PATH would kill all transcoding, not just tone mapping.
DIR=${0%/*}
CHAINED="$DIR/Plex Transcoder.preplacebo"
CUSTOM="${PLACEBO_TRANSCODER:-/plex-placebo/bin/Plex-Transcoder-placebo}"

# Every curve Plex's UI offers (linear/gamma/clip/reinhard/hable/mobius) exists in
# libplacebo under the same name, so the setting is carried across rather than overridden --
# the mod's job is to move tone mapping onto the GPU, not to second-guess the user's look.
# PLACEBO_TONEMAP forces a curve Plex can't select, e.g. bt.2390 or spline.
PLACEBO_CURVES='auto clip st2094-40 st2094-10 bt.2390 bt.2446a spline reinhard mobius hable gamma linear'

log() { [ -n "${PLACEBO_DEBUG:-}" ] && printf '%s\n' "$*" >> /tmp/plex-placebo.log; return 0; }

exec_custom() {
    # Bundled libplacebo/Vulkan/VA first, so we get the versions this binary was built
    # against rather than whatever the container ships. Any existing LD_LIBRARY_PATH (e.g.
    # from the vaapi-amdgpu mod) is kept after ours as a fallback.
    #
    # Ours does shadow that mod's libva. That's deliberate: its stack is musl-built and our
    # binary is glibc, so they can't be mixed -- we carry a matching glibc libva and VA
    # driver instead. If VAAPI encoding ever breaks after a container bump, look here first.
    export LD_LIBRARY_PATH="/plex-placebo/lib:${LD_LIBRARY_PATH:-}"
    export LIBVA_DRIVERS_PATH="/plex-placebo/lib/dri"
    export VK_DRIVER_FILES="/plex-placebo/icd.d/radeon_icd.x86_64.json"
    export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
    # Additive, so it can't hide the container's own layers. Only does anything when the
    # user sets MESA_VK_DEVICE_SELECT to pick between multiple AMD GPUs; ignored by loaders
    # too old to know the variable.
    export VK_ADD_IMPLICIT_LAYER_PATH="/plex-placebo/layer.d"
    export MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR:-/config/cache/placebo}"
    # Our build compiles h264/hevc in natively; Plex's codec blobs are musl-linked and
    # would fail to dlopen against a glibc binary.
    unset FFMPEG_EXTERNAL_LIBS
    # Terminal by design: once exec'd we cannot fall back, and a runtime failure (GPU busy,
    # /dev/dri perms, driver mismatch after a container bump) surfaces to Plex as a failed
    # transcode. Retrying via the chain would mean buffering output to know whether the
    # first attempt produced any, which is not worth the complexity for a case that means
    # the mod is misconfigured anyway.
    exec "$CUSTOM" "$@"
}

# Every fallback goes through here, so a missing chain target can't take out all
# transcoding: our own build is a complete ffmpeg and is a better last resort than exiting.
chain() {
    [ -x "$CHAINED" ] && exec "$CHAINED" "$@"
    log "chain target missing: $CHAINED"
    [ -x "$CUSTOM" ] && exec_custom "$@"
    echo "plex-transcoder-placebo: no usable transcoder found" >&2
    exit 127
}

case "$*" in
    *tonemap=*) ;;
    *) chain "$@" ;;
esac
[ -x "$CUSTOM" ] || { log "custom binary missing, chaining"; chain "$@"; }

# Find the filter graph up front. Everything below inspects it before any rewriting, so an
# unsuitable job can still chain with the original argv intact. Getting that ordering wrong
# is expensive: a bad graph only fails once libplacebo initialises, by which point we have
# exec'd and can no longer fall back.
graph=
prev=
for a in "$@"; do
    case $prev in
        -filter_complex|-vf|-filter:v|-filter:0)
            case $a in
                *tonemap=*) graph=$a; break ;;
            esac
            ;;
    esac
    prev=$a
done
[ -n "$graph" ] || { log "no tone map in a filter graph, chaining"; chain "$@"; }

# Inspect encoders and output count before rewriting anything, for the same reason the graph
# is found first: both only fail after exec, where the chain is gone.
prev=
formats=0
muxers=
segment_format=
encoder=
for a in "$@"; do
    case $prev in
        -codec|-codec:*|-c|-c:*|-vcodec|-acodec|-scodec)
            # The only external encoders configure enables are libopus and libvorbis, so any
            # other lib* -- libx264, libx265, libmp3lame, libvpx-vp9 -- is absent here and
            # would die as "Unknown encoder". Keep this pair in step with build.sh.
            case $a in
                libopus|libvorbis) ;;
                lib*) encoder=$a ;;
            esac ;;
        # ffmpeg groups every input ahead of every output, so anything collected before an -i
        # described that input: -f was its demuxer and -c its decoder, neither of which says
        # what this job encodes to or muxes into.
        -i) formats=0; muxers=; encoder= ;;
        -f)
            formats=$((formats + 1))
            muxers="$muxers $a" ;;
        -segment_format) segment_format=$a ;;
    esac
    prev=$a
done
[ -z "$encoder" ] || { log "encoder $encoder absent from this build; chaining"; chain "$@"; }

# Plex's ffmpeg 6.1 pulls only the filtergraph belonging to whichever output stream has the
# oldest timestamp, and a sparse subtitle output holds that spot for seconds at a time. The
# libplacebo graph then never gets pulled while frames keep arriving, and its queue grows
# past 4 GiB. Only the shape the patched build was tested against can opt back in.
if [ "$formats" -gt 1 ]; then
    has_dash=
    has_segment=
    case "$muxers " in *' dash '*) has_dash=1 ;; esac
    case "$muxers " in *' segment '*) has_segment=1 ;; esac
    if [ "${PLACEBO_EXPERIMENTAL_SUBTITLES:-}" = 1 ] && [ "$formats" -eq 2 ] &&
       [ -n "$has_dash" ] && [ -n "$has_segment" ] && [ "$segment_format" = ass ]; then
        log "experimental DASH + ASS subtitle job enabled"
    else
        log "multiple outputs require a supported opt-in; chaining"
        chain "$@"
    fi
fi

# Optional output-height gate. Unset means take every resolution; a value caps it. This is
# for hardware where tone mapping at 4K is slower on the GPU than Plex's software path -- true
# of small iGPUs -- so an operator there sets e.g. PLACEBO_MAX_HEIGHT=1080 to keep 4K jobs on
# the CPU. A dedicated GPU that can sustain 4K leaves it unset.
if [ -n "${PLACEBO_MAX_HEIGHT:-}" ]; then
    # Plex writes the output resolution into the graph as scale filters.
    max_h=0
    for h in $(printf '%s' "$graph" | grep -oE 'scale[^,;]*h=[0-9]+' | grep -oE '[0-9]+$') \
             $(printf '%s' "$graph" | grep -oE 'scale=[0-9]+:[0-9]+' | cut -d: -f2); do
        [ "$h" -gt "$max_h" ] && max_h=$h
    done
    if [ "$max_h" -eq 0 ]; then
        # No scale filter: the output is the source resolution, which argv doesn't reveal.
        # With a cap set we can't confirm the job is under it, so chain.
        log "no scale filter, cannot check PLACEBO_MAX_HEIGHT=$PLACEBO_MAX_HEIGHT, chaining"
        chain "$@"
    fi
    if [ "$max_h" -gt "$PLACEBO_MAX_HEIGHT" ]; then
        log "output height $max_h above PLACEBO_MAX_HEIGHT=$PLACEBO_MAX_HEIGHT, chaining"
        chain "$@"
    fi
fi

curve="${PLACEBO_TONEMAP:-}"
if [ -z "$curve" ]; then
    # Handles `tonemap=hable`, `tonemap=hable:desat=0` and Plex's
    # `tonemap=tonemap=hable:param=1.0` spelling.
    curve=$(printf '%s' "$graph" \
        | sed -nE 's/.*tonemap=(tonemap=)?([a-zA-Z0-9._-]+).*/\2/p')
fi

case " $PLACEBO_CURVES " in
    *" $curve "*) ;;
    *) log "curve '$curve' not supported by libplacebo, chaining"; chain "$@" ;;
esac

# PLACEBO_OPTS passes extra libplacebo options through, e.g.
#   PLACEBO_OPTS=contrast_recovery=0        less highlight clipping
#   PLACEBO_OPTS=peak_detect=0              no clipping at all, flatter image
# Restricted to option-ish characters: this string lands in a sed replacement, where a
# stray / or & would corrupt the whole filter graph. An option libplacebo rejects still
# kills the job, since that only surfaces after exec.
placebo_opts="${PLACEBO_OPTS:-}"
if [ -n "$placebo_opts" ]; then
    case $placebo_opts in
        *[!A-Za-z0-9_.:=-]*)
            log "PLACEBO_OPTS contains unsafe characters, ignoring: $placebo_opts"
            placebo_opts=
            ;;
    esac
fi

PLACEBO="libplacebo=tonemapping=$curve:${placebo_opts:+$placebo_opts:}colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv:format=nv12"

# Move the downscale onto the GPU and ahead of the tone map.
#
# Plex decodes on the GPU but omits -hwaccel_output_format, so every 4K frame is copied to
# system memory and scaled there. Tone mapping then costs whatever libplacebo is handed, and
# handing it 4K is four times the work of 1080p. Measured, 4K->1080p, 335 frames:
#
#   sw scale + sw tonemap (Plex today)      17.6s wall  32.9s cpu
#   sw scale + libplacebo (in-place swap)   17.6s wall  18.0s cpu
#   scale_vaapi then libplacebo at 1080p    11.6s wall   4.2s cpu
#
# Note it's the scale ordering that pays, not skipping the download: leaving the scale to
# libplacebo and only avoiding the copy still took 18.2s. Downloading AFTER the GPU scale is
# cheap (1080p nv12 is ~8x smaller than 4K p010) and keeps Plex's software subtitle overlay
# working untouched downstream.
#
# Only the video chain may become scale_vaapi; the subtitle scale operates on software ARGB
# and would fail. The video chain is found by following the label that feeds the tone map,
# rather than by guessing from how the scale happens to be spelled.
rewrite_graph() {
    rg_g=$1
    rg_tm=$(printf '%s' "$rg_g" | tr ';' '\n' | grep 'tonemap=' | head -1)
    rg_tm_in=$(printf '%s' "$rg_tm" | sed -nE 's/^\[([^]]+)\].*/\1/p')
    [ -n "$rg_tm_in" ] || return 1

    rg_src=$(printf '%s' "$rg_g" | tr ';' '\n' | grep -E "\[$rg_tm_in\]\$" | head -1)
    [ -n "$rg_src" ] || return 1
    case $rg_src in
        *scale=*) ;;
        *) return 1 ;;
    esac
    # A segment with more than one filter is not a plain scale; don't restructure blind.
    case $rg_src in
        *,*) return 1 ;;
    esac

    rg_in=$(printf '%s' "$rg_src" | sed -nE 's/^\[([^]]+)\].*/\1/p')
    [ -n "$rg_in" ] || return 1
    rg_w=$(printf '%s' "$rg_src" | sed -nE 's/.*scale=w=([0-9]+):h=([0-9]+).*/\1/p')
    rg_h=$(printf '%s' "$rg_src" | sed -nE 's/.*scale=w=([0-9]+):h=([0-9]+).*/\2/p')
    if [ -z "$rg_w" ]; then
        rg_w=$(printf '%s' "$rg_src" | sed -nE 's/.*scale=([0-9]+):([0-9]+).*/\1/p')
        rg_h=$(printf '%s' "$rg_src" | sed -nE 's/.*scale=([0-9]+):([0-9]+).*/\2/p')
    fi
    [ -n "$rg_w" ] && [ -n "$rg_h" ] || return 1

    rg_new_src="[$rg_in]scale_vaapi=w=$rg_w:h=$rg_h,hwdownload,format=p010le[$rg_tm_in]"
    rg_new_tm=$(printf '%s' "$rg_tm" \
        | sed -E "s/(format=(pix_fmts=)?p010,)?tonemap=[^],;[]+/$PLACEBO/g") || return 1
    [ "$rg_new_tm" != "$rg_tm" ] || return 1

    rg_out=
    rg_oldifs=$IFS
    IFS=';'
    for rg_seg in $rg_g; do
        if [ "$rg_seg" = "$rg_src" ]; then
            rg_seg=$rg_new_src
        elif [ "$rg_seg" = "$rg_tm" ]; then
            rg_seg=$rg_new_tm
        fi
        rg_out="${rg_out}${rg_out:+;}${rg_seg}"
    done
    IFS=$rg_oldifs
    printf '%s' "$rg_out"
}

zerocopy=
if [ -z "${PLACEBO_NO_ZEROCOPY:-}" ]; then
    new_graph=$(rewrite_graph "$graph") && [ -n "$new_graph" ] && zerocopy=1
fi
[ -n "$zerocopy" ] || log "graph not restructurable, falling back to in-place tonemap swap"

# Rebuild argv, rewriting only the value that follows a filter-graph flag. Rewriting any arg
# containing `tonemap=` would corrupt e.g. a title or a filename that happened to contain it.
matched=
added_hwfmt=
prev=
n=$#
while [ "$n" -gt 0 ]; do
    a=$1; shift
    n=$((n - 1))

    # The restructured graph expects VAAPI surfaces, so ask the decoder to keep them there.
    # Goes immediately before the first -i, where ffmpeg reads input options.
    if [ -n "$zerocopy" ] && [ -z "$added_hwfmt" ] && [ "$a" = "-i" ]; then
        set -- "$@" -hwaccel_output_format vaapi
        added_hwfmt=1
    fi

    case $prev in
        -filter_complex|-vf|-filter:v|-filter:0)
            case $a in
                *tonemap=*)
                    if [ -n "$zerocopy" ]; then
                        a=$new_graph
                    else
                        # [^,[;]* consumes the filter's own options too. Matching only the
                        # algorithm name would strip `tonemap=hable:desat=0` down to `hable`
                        # and leave `:desat=0` dangling on libplacebo, which rejects it and
                        # kills the job -- and by then `matched` has disabled the fallback.
                        a=$(printf '%s' "$a" \
                            | sed -E "s/(format=(pix_fmts=)?p010,)?tonemap=[^],;[]+/$PLACEBO/g") || a=$prev
                    fi
                    matched=1
                    ;;
            esac
            ;;
    esac
    prev=$a
    set -- "$@" "$a"
done

[ -n "$matched" ] || { log "no rewrite applied, chaining"; chain "$@"; }

log "rewrote: $*"
exec_custom "$@"
