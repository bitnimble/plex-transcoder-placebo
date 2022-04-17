## plex-transcoder-placebo

This project is a custom build of ffmpeg which contains both Plex's customisations as well as a backport of `libplacebo`. This allows for real-time shaders to be applied to any Plex video stream, to do things like realtime sharpening, enhancement, super resolution, etc.
The current goal is to get Anime4K's mpv shaders working in Plex. You can see a sample of what that looks like [here](https://youtu.be/1ZvxHlxTyow).

## Status

### April 2021
After pulling the latest PMS, it no longer works unfortunately. I've spent this time cleaning up the docker build scripts instead, which are now published [here](https://github.com/bitnimble/plex-transcoder-placebo-build).

Next up is focusing on figuring out why it crashes the Docker container when PMS loads the transcoder. Some notes:
 - It builds successfully
 - When testing the binary manually, it works fine (both on my Linux dev box as well as inside the Plex docker container)
 - When Plex starts up / tries to execute the transcoder, it crashes (presumably with a segfault). Not 100% sure why yet, but my suspicions is that either a binding or a codec being loaded from FFMPEG_EXTERNAL_LIBS is expecting a different release version now, so hopefully it's just a quick patch there.

### March 2021
Elden Ring came out, so no updates

### January 2021

Working, but very slow, and half of Plex's custom codecs (e.g. hardware accelerated h264 decoding) is broken. I'm 99% sure it's due to the crappy port that I made in the filter list code when rebasing.

## Developing / building it yourself
Want to help out?

### Requirements
- A development box that supports Docker, so pretty much anything
- Clone the [build scripts repo](https://github.com/bitnimble/plex-transcoder-placebo-build), and then clone this repo inside of that. The folder structure should look like this:
```
plex-transcoder-placebo-build/
├─ Dockerfile
├─ build_ffmpeg.sh
├─ build_musl_dependencies.sh
├─ build.sh
├─ ...
├─ plex-transcoder-placebo/
│  ├─ configure
│  ├─ Makefile
│  ├─ ...
```
- Run `./build.sh`
- Hopefully, you get a successful build and a new `target` directory in the root, which should contain `ffmpeg_g` and all of the libraries as well (`libavformat.so.58` etc).

I still have to do some work to pin the versions on a lot of the dependencies, so it's possible that even though it's currently working, it might not work for you. Please report any problems you find as a new Github Issue.

(Original FFmpeg readme below)

FFmpeg README
=============

FFmpeg is a collection of libraries and tools to process multimedia content
such as audio, video, subtitles and related metadata.

## Libraries

* `libavcodec` provides implementation of a wider range of codecs.
* `libavformat` implements streaming protocols, container formats and basic I/O access.
* `libavutil` includes hashers, decompressors and miscellaneous utility functions.
* `libavfilter` provides means to alter decoded audio and video through a directed graph of connected filters.
* `libavdevice` provides an abstraction to access capture and playback devices.
* `libswresample` implements audio mixing and resampling routines.
* `libswscale` implements color conversion and scaling routines.

## Tools

* [ffmpeg](https://ffmpeg.org/ffmpeg.html) is a command line toolbox to
  manipulate, convert and stream multimedia content.
* [ffplay](https://ffmpeg.org/ffplay.html) is a minimalistic multimedia player.
* [ffprobe](https://ffmpeg.org/ffprobe.html) is a simple analysis tool to inspect
  multimedia content.
* Additional small tools such as `aviocat`, `ismindex` and `qt-faststart`.

## Documentation

The offline documentation is available in the **doc/** directory.

The online documentation is available in the main [website](https://ffmpeg.org)
and in the [wiki](https://trac.ffmpeg.org).

### Examples

Coding examples are available in the **doc/examples** directory.

## License

FFmpeg codebase is mainly LGPL-licensed with optional components licensed under
GPL. Please refer to the LICENSE file for detailed information.

## Contributing

Patches should be submitted to the ffmpeg-devel mailing list using
`git format-patch` or `git send-email`. Github pull requests should be
avoided because they are not part of our review process and will be ignored.
