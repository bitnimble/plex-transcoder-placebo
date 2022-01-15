/*
 * OMX Video encoder
 * Copyright (C) 2011 Martin Storsjo
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "config.h"

#include <OMX_Core.h>
#include <OMX_Component.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "libavutil/avstring.h"
#include "libavutil/avutil.h"
#include "libavutil/common.h"
#include "libavutil/imgutils.h"
#include "libavutil/log.h"
#include "libavutil/opt.h"

#include "avcodec.h"
#include "h264.h"
#include "internal.h"
#include "pthread_internal.h"

#include "omx_common.h"

static av_cold int omx_component_init(AVCodecContext *avctx, const char *role)
{
    OMXCodecContext *s = avctx->priv_data;
    OMX_VIDEO_PARAM_PORTFORMATTYPE video_port_format = { 0 };
    OMX_VIDEO_PARAM_BITRATETYPE vid_param_bitrate = { 0 };
    OMX_PARAM_PORTDEFINITIONTYPE in_port_params = { 0 }, out_port_params = { 0 };
    OMX_ERRORTYPE err;
    OMX_INDEXTYPE index;
    int i;

    int ret = ff_omx_component_init(avctx, role, &in_port_params, &out_port_params);

    if (ret < 0)
        return ret;

    s->color_format = 0;
    for (i = 0; ; i++) {
        INIT_STRUCT(video_port_format);
        video_port_format.nIndex = i;
        video_port_format.nPortIndex = s->in_port;
        if (OMX_GetParameter(s->handle, OMX_IndexParamVideoPortFormat, &video_port_format) != OMX_ErrorNone)
            break;
        if (video_port_format.eColorFormat == OMX_COLOR_FormatYUV420SemiPlanar) {
/*        if (video_port_format.eColorFormat == OMX_COLOR_FormatYUV420Planar ||
            video_port_format.eColorFormat == OMX_COLOR_FormatYUV420PackedPlanar) {*/
            s->color_format = video_port_format.eColorFormat;
            break;
        }
    }
    if (s->color_format == 0) {
        av_log(avctx, AV_LOG_ERROR, "No supported pixel formats (%d formats available)\n", i);
        return AVERROR_UNKNOWN;
    }

    if (OMX_GetExtensionIndex(s->handle,
                              (OMX_STRING)"OMX.realtek.plex.index.se_memcpy",
                              &index) == OMX_ErrorNone) {
        OMX_GetParameter(s->handle, index, &s->copy_func);
    }

    in_port_params.bEnabled   = OMX_TRUE;
    in_port_params.bPopulated = OMX_FALSE;
    in_port_params.eDomain    = OMX_PortDomainVideo;

    in_port_params.format.video.pNativeRender         = NULL;
    in_port_params.format.video.bFlagErrorConcealment = OMX_FALSE;
    in_port_params.format.video.eColorFormat          = s->color_format;
    s->stride     = avctx->width;
    s->plane_size = avctx->height;
    // If specific codecs need to manually override the stride/plane_size,
    // that can be done here.
    in_port_params.format.video.nStride      = s->stride;
    in_port_params.format.video.nSliceHeight = s->plane_size;
    in_port_params.format.video.nFrameWidth  = avctx->width;
    in_port_params.format.video.nFrameHeight = avctx->height;
    if (avctx->framerate.den > 0 && avctx->framerate.num > 0)
        in_port_params.format.video.xFramerate = (1LL << 16) * avctx->framerate.num / avctx->framerate.den;
    else
        in_port_params.format.video.xFramerate = (1LL << 16) * avctx->time_base.den / avctx->time_base.num;

    err = OMX_SetParameter(s->handle, OMX_IndexParamPortDefinition, &in_port_params);
    CHECK(err);
    err = OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition, &in_port_params);
    CHECK(err);
    s->stride         = in_port_params.format.video.nStride;
    s->plane_size     = in_port_params.format.video.nSliceHeight;
    s->num_in_buffers = in_port_params.nBufferCountActual;

    err = OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition, &out_port_params);
    out_port_params.bEnabled   = OMX_TRUE;
    out_port_params.bPopulated = OMX_FALSE;
    out_port_params.eDomain    = OMX_PortDomainVideo;
    out_port_params.format.video.pNativeRender = NULL;
    out_port_params.format.video.nFrameWidth   = avctx->width;
    out_port_params.format.video.nFrameHeight  = avctx->height;
    out_port_params.format.video.nStride       = 0;
    out_port_params.format.video.nSliceHeight  = 0;
    out_port_params.format.video.nBitrate      = avctx->bit_rate;
    out_port_params.format.video.xFramerate    = in_port_params.format.video.xFramerate;
    out_port_params.format.video.bFlagErrorConcealment  = OMX_FALSE;
    if (avctx->codec->id == AV_CODEC_ID_MPEG4)
        out_port_params.format.video.eCompressionFormat = OMX_VIDEO_CodingMPEG4;
    else if (avctx->codec->id == AV_CODEC_ID_H264)
        out_port_params.format.video.eCompressionFormat = OMX_VIDEO_CodingAVC;

    err = OMX_SetParameter(s->handle, OMX_IndexParamPortDefinition, &out_port_params);
    CHECK(err);
    err = OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition, &out_port_params);
    CHECK(err);
    s->num_out_buffers = out_port_params.nBufferCountActual;

    INIT_STRUCT(vid_param_bitrate);
    vid_param_bitrate.nPortIndex     = s->out_port;
    vid_param_bitrate.eControlRate   = OMX_Video_ControlRateVariable;
    vid_param_bitrate.nTargetBitrate = avctx->bit_rate;
    err = OMX_SetParameter(s->handle, OMX_IndexParamVideoBitrate, &vid_param_bitrate);
    if (err != OMX_ErrorNone)
        av_log(avctx, AV_LOG_WARNING, "Unable to set video bitrate parameter\n");

    if (avctx->codec->id == AV_CODEC_ID_H264) {
        OMX_VIDEO_PARAM_AVCTYPE avc = { 0 };
        INIT_STRUCT(avc);
        avc.nPortIndex = s->out_port;
        err = OMX_GetParameter(s->handle, OMX_IndexParamVideoAvc, &avc);
        CHECK(err);
        avc.nBFrames = 0;
        avc.nPFrames = avctx->gop_size - 1;
        switch (s->profile == FF_PROFILE_UNKNOWN ? avctx->profile : s->profile) {
        case FF_PROFILE_H264_BASELINE:
            avc.eProfile = OMX_VIDEO_AVCProfileBaseline;
            break;
        case FF_PROFILE_H264_MAIN:
            avc.eProfile = OMX_VIDEO_AVCProfileMain;
            break;
        case FF_PROFILE_H264_HIGH:
            avc.eProfile = OMX_VIDEO_AVCProfileHigh;
            break;
        default:
            break;
        }
        err = OMX_SetParameter(s->handle, OMX_IndexParamVideoAvc, &avc);
        CHECK(err);
    }

    err = OMX_SendCommand(s->handle, OMX_CommandStateSet, OMX_StateIdle, NULL);
    CHECK(err);

    s->in_buffer_headers  = av_mallocz(sizeof(OMX_BUFFERHEADERTYPE*) * s->num_in_buffers);
    s->free_in_buffers    = av_mallocz(sizeof(OMX_BUFFERHEADERTYPE*) * s->num_in_buffers);
    s->out_buffer_headers = av_mallocz(sizeof(OMX_BUFFERHEADERTYPE*) * s->num_out_buffers);
    s->done_out_buffers   = av_mallocz(sizeof(OMX_BUFFERHEADERTYPE*) * s->num_out_buffers);
    if (!s->in_buffer_headers || !s->free_in_buffers || !s->out_buffer_headers || !s->done_out_buffers)
        return AVERROR(ENOMEM);
    for (i = 0; i < s->num_in_buffers && err == OMX_ErrorNone; i++) {
        if (s->input_zerocopy)
            err = OMX_UseBuffer(s->handle, &s->in_buffer_headers[i], s->in_port, s, in_port_params.nBufferSize, NULL);
        else
            err = OMX_AllocateBuffer(s->handle, &s->in_buffer_headers[i],  s->in_port,  s, in_port_params.nBufferSize);
        if (err == OMX_ErrorNone)
            s->in_buffer_headers[i]->pAppPrivate = s->in_buffer_headers[i]->pOutputPortPrivate = NULL;
    }
    CHECK(err);
    s->num_in_buffers = i;
    for (i = 0; i < s->num_out_buffers && err == OMX_ErrorNone; i++)
        err = OMX_AllocateBuffer(s->handle, &s->out_buffer_headers[i], s->out_port, s, out_port_params.nBufferSize);
    CHECK(err);
    s->num_out_buffers = i;

    if (ff_omx_wait_for_state(s, OMX_StateIdle) < 0) {
        av_log(avctx, AV_LOG_ERROR, "Didn't get OMX_StateIdle\n");
        return AVERROR_UNKNOWN;
    }
    err = OMX_SendCommand(s->handle, OMX_CommandStateSet, OMX_StateExecuting, NULL);
    CHECK(err);
    if (ff_omx_wait_for_state(s, OMX_StateExecuting) < 0) {
        av_log(avctx, AV_LOG_ERROR, "Didn't get OMX_StateExecuting\n");
        return AVERROR_UNKNOWN;
    }

    for (i = 0; i < s->num_out_buffers && err == OMX_ErrorNone; i++)
        err = OMX_FillThisBuffer(s->handle, s->out_buffer_headers[i]);
    if (err != OMX_ErrorNone) {
        for (; i < s->num_out_buffers; i++)
            s->done_out_buffers[s->num_done_out_buffers++] = s->out_buffer_headers[i];
    }
    for (i = 0; i < s->num_in_buffers; i++)
        s->free_in_buffers[s->num_free_in_buffers++] = s->in_buffer_headers[i];
    return err != OMX_ErrorNone ? AVERROR_UNKNOWN : 0;
}

static av_cold int omx_encode_init(AVCodecContext *avctx)
{
    OMXCodecContext *s = avctx->priv_data;
    const char *role;
    OMX_BUFFERHEADERTYPE *buffer;
    OMX_ERRORTYPE err;

    int ret = ff_omx_codec_init(avctx);
    if (ret < 0)
        return ret;

    switch (avctx->codec->id) {
    case AV_CODEC_ID_MPEG4:
        role = "video_encoder.mpeg4";
        break;
    case AV_CODEC_ID_H264:
        role = "video_encoder.avc";
        break;
    default:
        return AVERROR(ENOSYS);
    }

    if ((ret = omx_component_init(avctx, role)) < 0)
        goto fail;

    if (avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) {
        while (1) {
            buffer = ff_omx_get_buffer(s, &s->num_done_out_buffers, s->done_out_buffers, 1);
            if (buffer->nFlags & OMX_BUFFERFLAG_CODECCONFIG) {
                if ((ret = av_reallocp(&avctx->extradata, avctx->extradata_size + buffer->nFilledLen + AV_INPUT_BUFFER_PADDING_SIZE)) < 0) {
                    avctx->extradata_size = 0;
                    goto fail;
                }
                memcpy(avctx->extradata + avctx->extradata_size, buffer->pBuffer + buffer->nOffset, buffer->nFilledLen);
                avctx->extradata_size += buffer->nFilledLen;
                memset(avctx->extradata + avctx->extradata_size, 0, AV_INPUT_BUFFER_PADDING_SIZE);
            }
            err = OMX_FillThisBuffer(s->handle, buffer);
            if (err != OMX_ErrorNone) {
                ff_omx_append_buffer(s, &s->num_done_out_buffers, s->done_out_buffers, buffer);
                av_log(avctx, AV_LOG_ERROR, "OMX_FillThisBuffer failed: %x\n", err);
                ret = AVERROR_UNKNOWN;
                goto fail;
            }
            if (avctx->codec->id == AV_CODEC_ID_H264) {
                // For H.264, the extradata can be returned in two separate buffers
                // (the videocore encoder on raspberry pi does this);
                // therefore check that we have got both SPS and PPS before continuing.
                int nals[32] = { 0 };
                int i;
                for (i = 0; i + 4 < avctx->extradata_size; i++) {
                     if (!avctx->extradata[i + 0] &&
                         !avctx->extradata[i + 1] &&
                         !avctx->extradata[i + 2] &&
                         avctx->extradata[i + 3] == 1) {
                         nals[avctx->extradata[i + 4] & 0x1f]++;
                     }
                }
                if (nals[H264_NAL_SPS] && nals[H264_NAL_PPS])
                    break;
            } else {
                if (avctx->extradata_size > 0)
                    break;
            }
        }
    }

    return 0;
fail:
    return ret;
}

static void release_frame(void *opaque, uint8_t *data)
{
    AVFrame *frame = (void*)data;
    av_frame_free(&frame);
}

static int omx_encode_frame(AVCodecContext *avctx, AVPacket *pkt,
                            const AVFrame *frame, int *got_packet)
{
    OMXCodecContext *s = avctx->priv_data;
    int ret = 0;
    OMX_BUFFERHEADERTYPE* buffer;
    OMX_ERRORTYPE err;
    int had_partial = 0;

    if (frame) {
        uint8_t *dst[4];
        int linesize[4];
        int need_copy;
        buffer = ff_omx_get_buffer(s, &s->num_free_in_buffers, s->free_in_buffers, 1);

        buffer->nFilledLen = av_image_fill_arrays(dst, linesize, buffer->pBuffer, avctx->pix_fmt, s->stride, s->plane_size, 1);

        if (s->input_zerocopy) {
            uint8_t *src[4] = { NULL };
            int src_linesize[4];
            av_image_fill_arrays(src, src_linesize, frame->data[0], avctx->pix_fmt, s->stride, s->plane_size, 1);
            if (frame->linesize[0] == src_linesize[0] &&
                frame->linesize[1] == src_linesize[1] &&
                frame->linesize[2] == src_linesize[2] &&
                frame->data[1] == src[1] &&
                frame->data[2] == src[2]) {
                // If the input frame happens to have all planes stored contiguously,
                // with the right strides, just clone the frame and set the OMX
                // buffer header to point to it
                AVFrame *local = av_frame_clone(frame);
                AVBufferRef *buf = NULL;
                if (local)
                    buf = av_buffer_create((void*)local, sizeof(*local), release_frame, NULL, AV_BUFFER_FLAG_READONLY);
                if (!buf) {
                    // Return the buffer to the queue so it's not lost
                    ff_omx_append_buffer(s, &s->num_free_in_buffers, s->free_in_buffers, buffer);
                    return AVERROR(ENOMEM);
                } else {
                    buffer->pAppPrivate = local;
                    buffer->pOutputPortPrivate = NULL;
                    buffer->pBuffer = local->data[0];
                    need_copy = 0;
                }
            } else {
                // If not, we need to allocate a new buffer with the right
                // size and copy the input frame into it.
                AVBufferRef *buf = NULL;
                int image_buffer_size = av_image_get_buffer_size(avctx->pix_fmt, s->stride, s->plane_size, 1);
                if (image_buffer_size >= 0)
                    buf = av_buffer_alloc(image_buffer_size);
                if (!buf) {
                    // Return the buffer to the queue so it's not lost
                    ff_omx_append_buffer(s, &s->num_free_in_buffers, s->free_in_buffers, buffer);
                    return AVERROR(ENOMEM);
                } else {
                    buffer->pAppPrivate = buf;
                    buffer->pBuffer = buf->data;
                    need_copy = 1;
                    buffer->nFilledLen = av_image_fill_arrays(dst, linesize, buffer->pBuffer, avctx->pix_fmt, s->stride, s->plane_size, 1);
                }
            }
        } else {
            need_copy = 1;
        }
        if (need_copy) {
            if (s->copy_func && frame->data[3] && frame->buf[0]) {
                // Evil evil evil hack
                struct OMXOutputContext* octx = av_buffer_get_opaque(frame->buf[0]);
                need_copy = (s->copy_func(s->handle, buffer, octx->buffer, buffer->nFilledLen) != OMX_ErrorNone);
            }
        }
        if (need_copy)
            av_image_copy(dst, linesize, (const uint8_t**) frame->data, frame->linesize, avctx->pix_fmt, avctx->width, avctx->height);
        buffer->nFlags = OMX_BUFFERFLAG_ENDOFFRAME;
        buffer->nOffset = 0;
        // Convert the timestamps to microseconds; some encoders can ignore
        // the framerate and do VFR bit allocation based on timestamps.
        buffer->nTimeStamp = to_omx_ticks(av_rescale_q(frame->pts, avctx->time_base, AV_TIME_BASE_Q));
        if (frame->pict_type == AV_PICTURE_TYPE_I) {
#if CONFIG_OMX_RPI
            OMX_CONFIG_BOOLEANTYPE config = {0, };
            INIT_STRUCT(config);
            config.bEnabled = OMX_TRUE;
            err = OMX_SetConfig(s->handle, OMX_IndexConfigBrcmVideoRequestIFrame, &config);
            if (err != OMX_ErrorNone) {
                av_log(avctx, AV_LOG_ERROR, "OMX_SetConfig(RequestIFrame) failed: %x\n", err);
            }
#else
            OMX_CONFIG_INTRAREFRESHVOPTYPE config = {0, };
            INIT_STRUCT(config);
            config.nPortIndex = s->out_port;
            config.IntraRefreshVOP = OMX_TRUE;
            err = OMX_SetConfig(s->handle, OMX_IndexConfigVideoIntraVOPRefresh, &config);
            if (err != OMX_ErrorNone) {
                av_log(avctx, AV_LOG_ERROR, "OMX_SetConfig(IntraVOPRefresh) failed: %x\n", err);
            }
#endif
        }
        err = OMX_EmptyThisBuffer(s->handle, buffer);
        if (err != OMX_ErrorNone) {
            ff_omx_append_buffer(s, &s->num_free_in_buffers, s->free_in_buffers, buffer);
            av_log(avctx, AV_LOG_ERROR, "OMX_EmptyThisBuffer failed: %x\n", err);
            return AVERROR_UNKNOWN;
        }
    } else if (!s->eos_sent) {
        buffer = ff_omx_get_buffer(s, &s->num_free_in_buffers, s->free_in_buffers, 1);

        buffer->nFilledLen = 0;
        buffer->nFlags = OMX_BUFFERFLAG_EOS;
        buffer->pAppPrivate = buffer->pOutputPortPrivate = NULL;
        err = OMX_EmptyThisBuffer(s->handle, buffer);
        if (err != OMX_ErrorNone) {
            ff_omx_append_buffer(s, &s->num_free_in_buffers, s->free_in_buffers, buffer);
            av_log(avctx, AV_LOG_ERROR, "OMX_EmptyThisBuffer failed: %x\n", err);
            return AVERROR_UNKNOWN;
        }
        s->eos_sent = 1;
    }

    while (!*got_packet && ret == 0 && !s->got_eos) {
        // If not flushing, just poll the queue if there's finished packets.
        // If flushing, do a blocking wait until we either get a completed
        // packet, or get EOS.
        buffer = ff_omx_get_buffer(s, &s->num_done_out_buffers, s->done_out_buffers, !frame || had_partial);
        if (!buffer)
            break;

        if (buffer->nFlags & OMX_BUFFERFLAG_EOS)
            s->got_eos = 1;

        if (buffer->nFlags & OMX_BUFFERFLAG_CODECCONFIG && avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) {
            if ((ret = av_reallocp(&avctx->extradata, avctx->extradata_size + buffer->nFilledLen + AV_INPUT_BUFFER_PADDING_SIZE)) < 0) {
                avctx->extradata_size = 0;
                goto end;
            }
            memcpy(avctx->extradata + avctx->extradata_size, buffer->pBuffer + buffer->nOffset, buffer->nFilledLen);
            avctx->extradata_size += buffer->nFilledLen;
            memset(avctx->extradata + avctx->extradata_size, 0, AV_INPUT_BUFFER_PADDING_SIZE);
        } else {
            int newsize = s->output_buf_size + buffer->nFilledLen + AV_INPUT_BUFFER_PADDING_SIZE;
            if ((ret = av_reallocp(&s->output_buf, newsize)) < 0) {
                s->output_buf_size = 0;
                goto end;
            }
            memcpy(s->output_buf + s->output_buf_size, buffer->pBuffer + buffer->nOffset, buffer->nFilledLen);
            s->output_buf_size += buffer->nFilledLen;
            if (buffer->nFlags & OMX_BUFFERFLAG_ENDOFFRAME) {
                memset(s->output_buf + s->output_buf_size, 0, AV_INPUT_BUFFER_PADDING_SIZE);
                if ((ret = av_packet_from_data(pkt, s->output_buf, s->output_buf_size)) < 0) {
                    av_freep(&s->output_buf);
                    s->output_buf_size = 0;
                    goto end;
                }
                s->output_buf = NULL;
                s->output_buf_size = 0;
                pkt->pts = av_rescale_q(from_omx_ticks(buffer->nTimeStamp), AV_TIME_BASE_Q, avctx->time_base);
                // We don't currently enable B-frames for the encoders, so set
                // pkt->dts = pkt->pts. (The calling code behaves worse if the encoder
                // doesn't set the dts).
                pkt->dts = pkt->pts;
                if (buffer->nFlags & OMX_BUFFERFLAG_SYNCFRAME)
                    pkt->flags |= AV_PKT_FLAG_KEY;
                *got_packet = 1;
            } else {
#if CONFIG_OMX_RPI
                had_partial = 1;
#endif
            }
        }
end:
        err = OMX_FillThisBuffer(s->handle, buffer);
        if (err != OMX_ErrorNone) {
            ff_omx_append_buffer(s, &s->num_done_out_buffers, s->done_out_buffers, buffer);
            av_log(avctx, AV_LOG_ERROR, "OMX_FillThisBuffer failed: %x\n", err);
            ret = AVERROR_UNKNOWN;
        }
    }
    return ret;
}

static av_cold int omx_encode_end(AVCodecContext *avctx)
{
    OMXCodecContext *s = avctx->priv_data;

    ff_omx_cleanup(s);
    return 0;
}

#define OFFSET(x) offsetof(OMXCodecContext, x)
#define VDE AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM | AV_OPT_FLAG_ENCODING_PARAM
#define VE  AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM
static const AVOption options[] = {
    { "omx_libname", "OpenMAX library name", OFFSET(libname), AV_OPT_TYPE_STRING, { 0 }, 0, 0, VDE },
    { "omx_libprefix", "OpenMAX library prefix", OFFSET(libprefix), AV_OPT_TYPE_STRING, { 0 }, 0, 0, VDE },
    { "zerocopy", "Try to avoid copying input frames if possible", OFFSET(input_zerocopy), AV_OPT_TYPE_INT, { .i64 = CONFIG_OMX_RPI }, 0, 1, VE },
    { "profile",  "Set the encoding profile", OFFSET(profile), AV_OPT_TYPE_INT,   { .i64 = FF_PROFILE_UNKNOWN },       FF_PROFILE_UNKNOWN, FF_PROFILE_H264_HIGH, VE, "profile" },
    { "baseline", "",                         0,               AV_OPT_TYPE_CONST, { .i64 = FF_PROFILE_H264_BASELINE }, 0, 0, VE, "profile" },
    { "main",     "",                         0,               AV_OPT_TYPE_CONST, { .i64 = FF_PROFILE_H264_MAIN },     0, 0, VE, "profile" },
    { "high",     "",                         0,               AV_OPT_TYPE_CONST, { .i64 = FF_PROFILE_H264_HIGH },     0, 0, VE, "profile" },
    { NULL }
};

static const enum AVPixelFormat omx_encoder_pix_fmts[] = {
    AV_PIX_FMT_NV12, /* AV_PIX_FMT_YUV420P, */ AV_PIX_FMT_NONE
};

#define OMXENC(namev, longname, idv) \
static const AVClass omx_##namev##enc_class = { \
    .class_name = #namev "_omx", \
    .item_name  = av_default_item_name, \
    .option     = options, \
    .version    = LIBAVUTIL_VERSION_INT, \
}; \
const AVCodec ff_##namev##_omx_encoder = { \
    .name             = #namev "_omx", \
    .long_name        = NULL_IF_CONFIG_SMALL("OpenMAX IL " longname " video encoder"), \
    .type             = AVMEDIA_TYPE_VIDEO, \
    .id               = idv, \
    .priv_data_size   = sizeof(OMXCodecContext), \
    .init             = omx_encode_init, \
    .encode2          = omx_encode_frame, \
    .close            = omx_encode_end, \
    .pix_fmts         = omx_encoder_pix_fmts, \
    .capabilities     = AV_CODEC_CAP_DELAY, \
    .caps_internal    = FF_CODEC_CAP_INIT_THREADSAFE | FF_CODEC_CAP_INIT_CLEANUP, \
    .priv_class       = &omx_##namev##enc_class, \
};

OMXENC(mpeg4, "MPEG-4", AV_CODEC_ID_MPEG4)
OMXENC(h264, "H.264", AV_CODEC_ID_H264)
