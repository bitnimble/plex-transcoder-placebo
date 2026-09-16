#!/usr/bin/env python3
"""Exercise Plex/libplacebo sparse subtitles using software Vulkan only."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


def exercise(binary, root, fixture):
    root.mkdir()
    graph = ("[0:v]format=yuv420p10le,setparams=color_primaries=bt2020:"
             "color_trc=smpte2084:colorspace=bt2020nc,libplacebo=tonemapping=hable:"
             "colorspace=bt709:color_primaries=bt709:color_trc=bt709:format=yuv420p[v]")
    args = [binary, "-hide_banner", "-nostdin", "-loglevel", "info",
            "-init_hw_device", "vulkan=vk:0", "-filter_hw_device", "vk",
            "-filter_complex_threads", "2", "-threads", "2", "-i", str(fixture),
            "-filter_complex", graph, "-map", "[v]", "-map", "0:a",
            "-c:v", "rawvideo", "-threads:v", "2", "-c:a", "pcm_s16le",
            "-f", "null", "-", "-map", "0:s", "-c:s", "ass",
            "-f", "segment", "-segment_format", "ass", "-segment_time", "5",
            "-segment_list", str(root / "subs.list"), str(root / "subs%05d.ass"),
            "-progress", str(root / "progress.txt")]
    peak = 0
    deadline = time.monotonic() + 75
    with (root / "stderr.log").open("w") as log:
        process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=log)
        try:
            while process.poll() is None:
                try:
                    status = Path(f"/proc/{process.pid}/status").read_text()
                    rss = re.search(r"^VmRSS:\s+(\d+)", status, re.M)
                    if rss:
                        peak = max(peak, int(rss[1]))
                except FileNotFoundError:
                    pass
                if time.monotonic() > deadline or peak > 1536 * 1024:
                    raise RuntimeError("Software test exceeded time or memory budget")
                time.sleep(0.2)
            if process.returncode:
                raise RuntimeError((root / "stderr.log").read_text()[-3000:])
        finally:
            # No timeout: this runs while an exception may be propagating, and a raise here
            # would replace the budget or exit-status failure with a bare TimeoutExpired.
            if process.poll() is None:
                process.kill()
            process.wait()
    progress = (root / "progress.txt").read_text()
    frames = re.findall(r"^frame=(\d+)$", progress, re.M)
    subtitles = "\n".join(p.read_text() for p in root.glob("subs*.ass"))
    if not frames or int(frames[-1]) != 2880 or "progress=end" not in progress:
        raise RuntimeError("Expected all 2880 video frames and a completed job")
    if "Early caption" not in subtitles or "Late caption" not in subtitles:
        raise RuntimeError("Missing sparse subtitle content")
    return {"frames": int(frames[-1]), "peak_rss_mib": round(peak / 1024, 1),
            "subtitle_segments": len(list(root.glob("subs*.ass")))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--icd", required=True, help="Mesa lavapipe ICD JSON")
    args = parser.parse_args()
    # A software ICD selection alone is not a boundary against host GPU faults.
    if Path("/dev/dri").exists() or Path("/dev/kfd").exists():
        parser.error("Run in a resource-limited container without GPU device mappings")
    icd = Path(args.icd)
    if "lvp" not in icd.name or not icd.is_file():
        parser.error("An existing lavapipe (lvp) ICD file is required")
    os.environ.update(VK_ICD_FILENAMES=str(icd), VK_DRIVER_FILES=str(icd),
                      XDG_RUNTIME_DIR="/tmp", LP_NUM_THREADS="2")
    candidate = str(Path(args.candidate).resolve(strict=True))
    with tempfile.TemporaryDirectory(prefix="plex-placebo-software-") as tmp:
        root = Path(tmp)
        subs = root / "sparse.srt"
        subs.write_text("1\n00:00:01,000 --> 00:00:02,000\nEarly caption\n\n"
                        "2\n00:01:50,000 --> 00:01:51,000\nLate caption\n")
        fixture = root / "fixture.mkv"
        subprocess.run([candidate, "-hide_banner", "-nostdin", "-loglevel", "error",
                        "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24:duration=120",
                        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=120",
                        "-i", str(subs), "-map", "0:v", "-map", "1:a", "-map", "2:s",
                        "-c:v", "mpeg4", "-threads:v", "2", "-q:v", "8",
                        "-c:a", "pcm_s16le", "-c:s", "srt", str(fixture)],
                       check=True, timeout=45)
        results = {}
        if args.baseline:
            results["baseline"] = exercise(str(Path(args.baseline).resolve(strict=True)),
                                            root / "baseline", fixture)
        results["candidate"] = exercise(candidate, root / "candidate", fixture)
        print(json.dumps(results, indent=2), flush=True)
        # The unaffected case stays far below this limit on the pinned software stack.
        if results["candidate"]["peak_rss_mib"] > 512:
            raise RuntimeError("Candidate retained too much memory across sparse subtitles")


if __name__ == "__main__":
    main()
