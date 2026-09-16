#!/bin/bash
# Tests the filter-graph rewrite: which curve comes out, and that malformed or unsupported
# input chains instead of producing a graph that only fails after exec (at which point the
# wrapper can no longer fall back, and Plex sees a dead transcode).
#
# If an ffmpeg with libplacebo is on PATH, the in-place (non-zerocopy) rewritten graphs are
# replayed through it to prove they open. The zerocopy graphs (scale_vaapi + hwdownload) can
# only be checked structurally here -- replaying them needs a working VAAPI device, which CI
# lacks; that path is validated on real hardware.
set -uo pipefail
REPO=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
fail=0

install -m755 "$REPO/plex-transcoder-wrapper.sh" "$T/Plex Transcoder"
printf '#!/bin/sh\necho CHAINED\n' > "$T/Plex Transcoder.preplacebo"
printf '#!/bin/sh\nprintf "%%s\\n" "$@"\n' > "$T/custom"
chmod +x "$T/Plex Transcoder.preplacebo" "$T/custom"

# No pipe: `... | grep -q` would SIGPIPE ffmpeg, and pipefail would turn that into "no
# libplacebo", silently skipping the graph replay that is the point of this suite.
FF=""
if command -v ffmpeg >/dev/null; then
    filters=$(ffmpeg -hide_banner -filters 2>/dev/null || true)
    case $filters in *libplacebo*) FF=ffmpeg;; esac
fi

# Mirrors Plex's real graph shape, including the scale that encodes the output resolution.
PRE='[1:0]scale=64:64[ov];[0:0]scale=w=320:h=180[1];[1]'
POST='[2];[2]format=pix_fmts=nv12[3];[3][ov]overlay[out]'

# Returns the rewritten argv, or the literal CHAINED.
#
# Zero-copy is disabled here on purpose: these cases test curve selection and option
# handling, and the restructured graph needs a real VAAPI device to replay, which CI has
# not got. The restructure has its own section below.
run_wrapper() {
    local graph=$1 override=${2:-}
    if [ -n "$override" ]; then
        PLACEBO_TRANSCODER="$T/custom" PLACEBO_NO_ZEROCOPY=1 PLACEBO_TONEMAP="$override" \
            "$T/Plex Transcoder" -filter_complex "$graph" 2>&1
    else
        PLACEBO_TRANSCODER="$T/custom" PLACEBO_NO_ZEROCOPY=1 \
            "$T/Plex Transcoder" -filter_complex "$graph" 2>&1
    fi
}

expect_curve() {
    local label=$1 graph=$2 want=$3 override=${4:-} out curve
    out=$(run_wrapper "$graph" "$override")
    case $out in
        *CHAINED*) echo "FAIL [$label]: chained, expected tonemapping=$want"; fail=1; return;;
    esac
    curve=$(printf '%s' "$out" | sed -nE 's/.*tonemapping=([^:]+).*/\1/p')
    [ "$curve" = "$want" ] || { echo "FAIL [$label]: got tonemapping=$curve, want $want"; fail=1; return; }
    if [ -n "$FF" ]; then
        local g; g=$(printf '%s' "$out" | sed -n '2p')
        "$FF" -hide_banner -f lavfi -i testsrc2=s=320x180:d=1 -f lavfi -i color=c=red:s=64x64:d=1 \
            -filter_complex "$g" -map '[out]' -frames:v 1 -f null - >/dev/null 2>&1 \
            || { echo "FAIL [$label]: rewritten graph does not open"; fail=1; return; }
    fi
    echo "  ok: $label -> $curve"
}

expect_chain() {
    local label=$1 graph=$2 out
    out=$(run_wrapper "$graph")
    case $out in
        *CHAINED*) echo "  ok: $label -> chained";;
        *) echo "FAIL [$label]: rewrote instead of chaining ($out)"; fail=1;;
    esac
}

# Plex's UI offers exactly these; the setting must survive, not be overridden.
for c in linear gamma clip reinhard hable mobius; do
    expect_curve "plex curve $c" "${PRE}format=p010,tonemap=$c${POST}" "$c"
done

# Curves Plex can't select but libplacebo supports.
for c in bt.2390 spline bt.2446a st2094-40; do
    expect_curve "override $c" "${PRE}format=p010,tonemap=hable${POST}" "$c" "$c"
done

# The tonemap filter's own options must not leak onto libplacebo's option list.
expect_curve "options: desat"  "${PRE}format=p010,tonemap=hable:desat=0${POST}" hable
expect_curve "options: param"  "${PRE}format=p010,tonemap=tonemap=mobius:param=1.0${POST}" mobius
expect_curve "no format prefix" "${PRE}tonemap=reinhard${POST}" reinhard
expect_curve "pix_fmts spelling" "${PRE}format=pix_fmts=p010,tonemap=hable${POST}" hable

# Unknown curve must chain: an invalid tonemapping= value only fails once libplacebo
# initialises, long after the wrapper has exec'd and lost its chance to fall back.
expect_chain "unsupported curve" "${PRE}format=p010,tonemap=nosuchcurve${POST}"
expect_chain "no tonemap at all" "${PRE}scale=64:64${POST}"

# Resolution gate. Off by default (every resolution processed); PLACEBO_MAX_HEIGHT opts into
# a cap, for hardware where 4K tone mapping is slower on the GPU than in software.
G4K='[1:0]scale=3840:2160[ov];[0:0]scale=w=3840:h=2160:force_divisible_by=4[1];[1]format=p010,tonemap=hable[2];[2]format=pix_fmts=nv12[3];[3][ov]overlay[out]'
G1440='[0:0]scale=w=2560:h=1440[1];[1]format=p010,tonemap=hable[2];[2]format=pix_fmts=nv12[out]'
GNOSCALE='[0:0]format=p010,tonemap=hable[out]'

gate() { # label want-verdict env-assignments... graph-is-last-arg
    local label=$1 want=$2; shift 2
    local graph; eval "graph=\${$#}"
    local out
    out=$(env "$@" PLACEBO_TRANSCODER="$T/custom" PLACEBO_NO_ZEROCOPY=1 \
          "$T/Plex Transcoder" -hwaccel:0 vaapi -i x.mkv -filter_complex "$graph" 2>&1)
    local got=chained
    case $out in *libplacebo=*) got=accelerated;; esac
    if [ "$got" = "$want" ]; then
        echo "  ok: $label -> $got"
    else
        echo "FAIL [$label]: got $got, want $want"; fail=1
    fi
}

# Default: unset means take everything, 4K included.
gate "default takes 4K"          accelerated "$G4K"
gate "default takes no-scale"    accelerated "$GNOSCALE"
# Capped at 1080: 4K and 1440p chain, 1080p and below are taken.
gate "cap 1080 skips 4K"         chained     PLACEBO_MAX_HEIGHT=1080 "$G4K"
gate "cap 1080 skips 1440p"      chained     PLACEBO_MAX_HEIGHT=1080 "$G1440"
gate "cap 1080 takes 1080p"      accelerated PLACEBO_MAX_HEIGHT=1080 \
    '[0:0]scale=w=1920:h=1080[1];[1]format=p010,tonemap=hable[2];[2]format=pix_fmts=nv12[out]'
# With a cap set, a scale-less graph can't be checked, so it chains.
gate "cap skips no-scale"        chained     PLACEBO_MAX_HEIGHT=1080 "$GNOSCALE"
# A higher cap lets a bigger GPU take 4K.
gate "cap 2160 takes 4K"         accelerated PLACEBO_MAX_HEIGHT=2160 "$G4K"

expect_curve "720p output" \
    '[0:0]scale=w=1280:h=720[1];[1]format=p010,tonemap=mobius[2];[2]format=pix_fmts=nv12[out]' mobius

# Args that merely contain the text must not be touched.
out=$(PLACEBO_TRANSCODER="$T/custom" "$T/Plex Transcoder" \
      -metadata "title=learning tonemap=hable" \
      -filter_complex "${PRE}format=p010,tonemap=hable${POST}" 2>&1)
if printf '%s' "$out" | grep -q 'title=learning tonemap=hable'; then
    echo "  ok: unrelated arg preserved"
else
    echo "FAIL: -metadata value was rewritten"; fail=1
fi

# PLACEBO_OPTS lands in a sed replacement, so anything outside option characters has to be
# rejected rather than allowed to corrupt the filter graph.
opts_out=$(PLACEBO_TRANSCODER="$T/custom" PLACEBO_NO_ZEROCOPY=1 PLACEBO_OPTS=contrast_recovery=0 \
    "$T/Plex Transcoder" -filter_complex "${PRE}format=p010,tonemap=hable${POST}" 2>&1)
case $opts_out in
    *contrast_recovery=0:colorspace=bt709*) echo "  ok: PLACEBO_OPTS injected";;
    *) echo "FAIL: PLACEBO_OPTS not injected"; fail=1;;
esac
bad_out=$(PLACEBO_TRANSCODER="$T/custom" PLACEBO_NO_ZEROCOPY=1 PLACEBO_OPTS='a=1/b&c' \
    "$T/Plex Transcoder" -filter_complex "${PRE}format=p010,tonemap=hable${POST}" 2>&1)
case $bad_out in
    *'a=1/b&c'*) echo "FAIL: unsafe PLACEBO_OPTS reached the graph"; fail=1;;
    *libplacebo=tonemapping=hable:colorspace*) echo "  ok: unsafe PLACEBO_OPTS rejected, graph intact";;
    *) echo "FAIL: unsafe PLACEBO_OPTS broke the rewrite"; fail=1;;
esac

# --- GPU-side scaling restructure -------------------------------------------------------
#
# Plex's own graph downloads 4K frames and scales in software, then tone maps at 4K. Moving
# the scale to the GPU and ahead of the tone map measured 11.6s wall / 4.2s cpu against
# 17.6s / 32.9s for Plex's shape. Only the video chain may be converted: the subtitle scale
# works on software ARGB and would fail as scale_vaapi.
PLEX_GRAPH='[0:2]scale=1920:1080[0];[0:0]scale=w=1920:h=1080:force_divisible_by=4[1];[1]format=p010,tonemap=hable[2];[2]format=pix_fmts=nv12[3];[3][0]overlay[4];[4]hwupload[5]'

zc=$(PLACEBO_TRANSCODER="$T/custom" "$T/Plex Transcoder" \
     -hwaccel:0 vaapi -i /tmp/in.mkv -filter_complex "$PLEX_GRAPH" -map '[5]' 2>&1)

check_zc() {
    if printf '%s' "$zc" | grep -q -- "$2"; then
        echo "  ok: $1"
    else
        echo "FAIL [zerocopy]: $1"; fail=1
    fi
}
check_zc "-hwaccel_output_format injected"  '-hwaccel_output_format'
check_zc "video scale moved to GPU"         'scale_vaapi=w=1920:h=1080,hwdownload,format=p010le'
check_zc "subtitle scale left in software"  '\[0:2\]scale=1920:1080\[0\]'
check_zc "curve preserved through rewrite"  'libplacebo=tonemapping=hable'
check_zc "overlay preserved"                'overlay'

# The escape hatch must fall back to the in-place swap, still replacing the tone map.
zc_off=$(PLACEBO_TRANSCODER="$T/custom" PLACEBO_NO_ZEROCOPY=1 "$T/Plex Transcoder" \
         -hwaccel:0 vaapi -i /tmp/in.mkv -filter_complex "$PLEX_GRAPH" -map '[5]' 2>&1)
case $zc_off in
    *scale_vaapi*) echo "FAIL [zerocopy]: PLACEBO_NO_ZEROCOPY did not disable restructure"; fail=1;;
    *libplacebo=*) echo "  ok: PLACEBO_NO_ZEROCOPY falls back to in-place swap";;
    *) echo "FAIL [zerocopy]: escape hatch lost the rewrite entirely"; fail=1;;
esac

# A video segment carrying more than a bare scale is not safe to restructure; it must fall
# back to the in-place swap rather than emit a graph that only fails after exec.
ODD='[0:0]scale=w=1920:h=1080,setsar=1[1];[1]format=p010,tonemap=hable[2];[2]format=pix_fmts=nv12[out]'
odd_out=$(PLACEBO_TRANSCODER="$T/custom" "$T/Plex Transcoder" \
          -hwaccel:0 vaapi -i /tmp/in.mkv -filter_complex "$ODD" 2>&1)
case $odd_out in
    *scale_vaapi*) echo "FAIL [zerocopy]: restructured a graph it should not have"; fail=1;;
    *libplacebo=*) echo "  ok: unrecognised video segment falls back to in-place swap";;
    *) echo "FAIL [zerocopy]: lost the rewrite on an unrecognised segment"; fail=1;;
esac

# An encoder the build lacks, and a second output, both only fail after exec, so both have
# to be caught here. A separate ASS subtitle output needs the explicit opt-in on top.
for flag in -codec:0 -c:v -codec -c -vcodec; do
    for codec in libx264 libx265 libmp3lame libvpx-vp9; do
        guarded=$(PLACEBO_TRANSCODER="$T/custom" "$T/Plex Transcoder" \
            -filter_complex "$PLEX_GRAPH" "$flag" "$codec" -crf:0 16 2>&1)
        case $guarded in
            *CHAINED*) echo "  ok: $flag $codec remains on stock transcoder";;
            *) echo "FAIL: absent encoder $codec entered custom build"; fail=1;;
        esac
    done
done
for codec in libopus libvorbis; do
    built=$(PLACEBO_TRANSCODER="$T/custom" "$T/Plex Transcoder" \
        -hwaccel:0 vaapi -i movie.mkv -filter_complex "$PLEX_GRAPH" \
        -codec:0 h264_vaapi -codec:1 "$codec" -f dash dash 2>&1)
    case $built in
        *libplacebo=*) echo "  ok: $codec is configured in, job still rewritten";;
        *) echo "FAIL: chained on $codec, which this build has"; fail=1;;
    esac
done

subs=$(PLACEBO_TRANSCODER="$T/custom" "$T/Plex Transcoder" \
    -hwaccel:0 vaapi -i movie.mkv -filter_complex "$PLEX_GRAPH" \
    -codec:0 h264_vaapi -f dash dash -map 0:3 -codec:0 ass \
    -f segment -segment_format ass 'sub-chunk-%05d' 2>&1)
case $subs in
    *CHAINED*) echo "  ok: separate subtitle output remains on stock transcoder";;
    *) echo "FAIL: subtitle segment job entered custom build"; fail=1;;
esac
safe=$(PLACEBO_TRANSCODER="$T/custom" "$T/Plex Transcoder" \
    -hwaccel:0 vaapi -i segment -filter_complex "$PLEX_GRAPH" \
    -metadata title=libx264 -codec:0 h264_vaapi -f dash dash 2>&1)
case $safe in
    *libplacebo=*) echo "  ok: VAAPI job and incidental names still rewritten";;
    *) echo "FAIL: safe VAAPI job incorrectly chained"; fail=1;;
esac

# Plex muxes its primary HLS output with the segment muxer: one output, no subtitle stream,
# nothing for the graph to starve on. Chaining it would drop the mod for every HLS client.
hls=$(PLACEBO_TRANSCODER="$T/custom" "$T/Plex Transcoder" \
    -hwaccel:0 vaapi -i movie.mkv -filter_complex "$PLEX_GRAPH" -codec:0 h264_vaapi \
    -f segment -segment_format mpegts -segment_time 5 'media-%05d.ts' 2>&1)
case $hls in
    *libplacebo=*) echo "  ok: single-output HLS segment job still rewritten";;
    *) echo "FAIL: single-output HLS job chained"; fail=1;;
esac

# An -f ahead of an -i names that input's demuxer, not an output.
demux=$(PLACEBO_TRANSCODER="$T/custom" "$T/Plex Transcoder" \
    -f matroska -i movie.mkv -filter_complex "$PLEX_GRAPH" -codec:0 h264_vaapi \
    -f dash dash 2>&1)
case $demux in
    *libplacebo=*) echo "  ok: input demuxer -f is not counted as an output";;
    *) echo "FAIL: input -f counted as a second output"; fail=1;;
esac

# Same trap for codecs: a -c ahead of an -i picks that input's decoder. libdav1d is a
# decoder this build has and has no encoder at all, so reading it as one chains a job that
# would have accelerated fine.
for dec in libdav1d libvpx-vp9; do
    decode=$(PLACEBO_TRANSCODER="$T/custom" "$T/Plex Transcoder" \
        -c:v "$dec" -i movie.mkv -filter_complex "$PLEX_GRAPH" \
        -codec:0 h264_vaapi -f dash dash 2>&1)
    case $decode in
        *libplacebo=*) echo "  ok: input decoder -c:v $dec is not read as an encoder";;
        *) echo "FAIL: input decoder $dec chained the job"; fail=1;;
    esac
done

# Burn-in is one output: PLEX_GRAPH overlays the subtitle stream, so it is rewritten like
# any other single-output job.
burn=$(PLACEBO_TRANSCODER="$T/custom" "$T/Plex Transcoder" \
    -hwaccel:0 vaapi -i movie.mkv -filter_complex "$PLEX_GRAPH" \
    -codec:0 h264_vaapi -f dash dash 2>&1)
case $burn in
    *libplacebo=*) echo "  ok: subtitle burn-in stays on the custom build";;
    *) echo "FAIL: single-output burn-in job chained"; fail=1;;
esac
second=$(PLACEBO_TRANSCODER="$T/custom" "$T/Plex Transcoder" \
    -filter_complex "$PLEX_GRAPH" -codec:0 h264_vaapi -f dash dash \
    -map 0:3 -f null - 2>&1)
case $second in
    *CHAINED*) echo "  ok: unrecognised second output remains on stock";;
    *) echo "FAIL: multiple-output job entered custom build"; fail=1;;
esac

for enabled in 1 0 true; do
    subs=$(PLACEBO_EXPERIMENTAL_SUBTITLES="$enabled" PLACEBO_TRANSCODER="$T/custom" \
        "$T/Plex Transcoder" -hwaccel:0 vaapi -i movie.mkv \
        -filter_complex "$PLEX_GRAPH" -codec:0 h264_vaapi -f dash dash \
        -map 0:3 -codec:0 ass -f segment -segment_format ass 'sub-%05d' 2>&1)
    case "$enabled:$subs" in
        1:*libplacebo=*|0:*CHAINED*|true:*CHAINED*)
            echo "  ok: separate ASS subtitles opt-in=$enabled" ;;
        *) echo "FAIL: unexpected subtitle opt-in result: $enabled"; fail=1 ;;
    esac
done
# Plex's argv order is not a promise, so the opt-in matches the set of muxers, not a string.
reordered=$(PLACEBO_EXPERIMENTAL_SUBTITLES=1 PLACEBO_TRANSCODER="$T/custom" \
    "$T/Plex Transcoder" -hwaccel:0 vaapi -i movie.mkv -filter_complex "$PLEX_GRAPH" \
    -map 0:3 -codec:0 ass -f segment -segment_format ass 'sub-%05d' \
    -codec:0 h264_vaapi -f dash dash 2>&1)
case $reordered in
    *libplacebo=*) echo "  ok: opt-in accepts the subtitle output first";;
    *) echo "FAIL: opt-in is sensitive to output order"; fail=1;;
esac
for codec in libx264 libx265 libmp3lame; do
    guarded=$(PLACEBO_EXPERIMENTAL_SUBTITLES=1 PLACEBO_TRANSCODER="$T/custom" \
        "$T/Plex Transcoder" -filter_complex "$PLEX_GRAPH" -codec:0 "$codec" \
        -f dash dash -f segment -segment_format ass 'sub-%05d' 2>&1)
    case $guarded in
        *CHAINED*) echo "  ok: opt-in cannot bypass the $codec guard" ;;
        *) echo "FAIL: opt-in bypassed the absent-encoder guard"; fail=1 ;;
    esac
done
for secondary in null segment; do
    guarded=$(PLACEBO_EXPERIMENTAL_SUBTITLES=1 PLACEBO_TRANSCODER="$T/custom" \
        "$T/Plex Transcoder" -filter_complex "$PLEX_GRAPH" -codec:0 h264_vaapi \
        -f dash dash -f "$secondary" -segment_format srt 'sub-%05d' 2>&1)
    case $guarded in
        *CHAINED*) echo "  ok: opt-in rejects untested $secondary/srt output" ;;
        *) echo "FAIL: opt-in accepted unsupported output"; fail=1 ;;
    esac
done
third=$(PLACEBO_EXPERIMENTAL_SUBTITLES=1 PLACEBO_TRANSCODER="$T/custom" \
    "$T/Plex Transcoder" -filter_complex "$PLEX_GRAPH" -codec:0 h264_vaapi \
    -f dash dash -f segment -segment_format ass 'sub-%05d' -f null - 2>&1)
case $third in
    *CHAINED*) echo "  ok: opt-in rejects a third output";;
    *) echo "FAIL: opt-in accepted a third output"; fail=1;;
esac

[ -n "$FF" ] || echo "  (note: no libplacebo ffmpeg on PATH; graphs not replayed)"
[ "$fail" = 0 ] && echo "PASS: all rewrite scenarios"
exit "$fail"
