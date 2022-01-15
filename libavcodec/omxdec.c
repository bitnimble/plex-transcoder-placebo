/*
 * OMX Video decoder
 * Copyright (C) 2002-2021 Realtek Semiconductor Corp.
 * Copyright (C) 2021 rcombs
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

#include "libavutil/avassert.h"

#include "decode.h"

#include "omx_common.h"

static int omx_reconfigure_output_buffer(AVCodecContext *avctx)
{
    OMXCodecContext *s = avctx->priv_data;
    OMX_PARAM_PORTDEFINITIONTYPE out_port_params = { 0 };
    OMX_ERRORTYPE err = OMX_ErrorNone;
    int ret = 0;
    int i = 0;

    if (s->out_buffer_headers && s->num_out_buffers)
    {
        int freed = 0;
        ret = OMX_SendCommand(s->handle, OMX_CommandPortDisable, s->out_port, NULL);
        if (ret != OMX_ErrorNone) {
            av_log(s->avctx, AV_LOG_ERROR, "OMX_SendCommand OMX_CommandPortDisable %d Error\n", __LINE__);
            return AVERROR_UNKNOWN;
        }

        pthread_mutex_lock(&s->handlectx->mutex);
        s->output_reconfig_count++;
        while (s->out_port_state || s->num_done_out_buffers) {
            OMX_BUFFERHEADERTYPE *buffer = ff_omx_get_buffer_locked(s, &s->num_done_out_buffers, s->done_out_buffers, 0);
            if (buffer) {
                if ((OMX_FreeBuffer(s->handle, s->out_port, buffer)) != OMX_ErrorNone)
                    av_log(s->avctx, AV_LOG_ERROR, "OMX_FreeBuffer %d Error\n", __LINE__);
                else
                    freed++;
            }

            if (s->out_port_state && !s->num_done_out_buffers)
                pthread_cond_wait(&s->handlectx->cond, &s->handlectx->mutex);
        }
        pthread_mutex_unlock(&s->handlectx->mutex);

        free(s->out_buffer_headers);
        free(s->done_out_buffers);
        s->out_buffer_headers = s->done_out_buffers = NULL;

        av_log(s->avctx, AV_LOG_DEBUG, "Freed %i of %i output buffers\n", freed, s->num_out_buffers);

        // Can't handle if this overflows u64 somehow
        if (s->output_reconfig_count == 0)
            return AVERROR(EINVAL);
    }

    INIT_STRUCT(out_port_params);
    out_port_params.nPortIndex = s->out_port;
    ret = OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition, &out_port_params);
    if (ret != OMX_ErrorNone) {
        av_log(s->avctx, AV_LOG_ERROR, "OMX_SetParameter OMX_IndexParamPortDefinition %d Error\n", __LINE__);
        return AVERROR_UNKNOWN;
    }

    out_port_params.bEnabled     = OMX_TRUE;
    out_port_params.bPopulated   = OMX_FALSE;
    out_port_params.eDomain      = OMX_PortDomainVideo;
    out_port_params.format.video.pNativeRender = NULL;

    ret = OMX_SetParameter(s->handle, OMX_IndexParamPortDefinition, &out_port_params);
    if (ret != OMX_ErrorNone) {
        av_log(s->avctx, AV_LOG_ERROR, "OMX_SetParameter OMX_IndexParamPortDefinition %d Error\n", __LINE__);
        return AVERROR_UNKNOWN;
    }

    ret = OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition, &out_port_params);
    if (ret != OMX_ErrorNone) {
        av_log(s->avctx, AV_LOG_ERROR, "OMX_SetParameter OMX_IndexParamPortDefinition %d Error\n", __LINE__);
        return AVERROR_UNKNOWN;
    }

    s->num_out_buffers     = out_port_params.nBufferCountActual;
    s->stride              = out_port_params.format.video.nStride;
    s->plane_size          = out_port_params.format.video.nSliceHeight;
    s->out_buffer_headers  = av_mallocz(sizeof(OMX_BUFFERHEADERTYPE*) * s->num_out_buffers);
    s->done_out_buffers    = av_mallocz(sizeof(OMX_BUFFERHEADERTYPE*) * s->num_out_buffers);

    if (!s->out_buffer_headers || !s->done_out_buffers)
        return AVERROR(ENOMEM);

    for (i = 0; i < s->num_out_buffers && err == OMX_ErrorNone; i++)
        err = OMX_AllocateBuffer(s->handle, &s->out_buffer_headers[i], s->out_port, s, out_port_params.nBufferSize);
    CHECK(err);

    pthread_mutex_lock(&s->handlectx->mutex);
    s->output_pending_reconfigure = 0;
    pthread_mutex_unlock(&s->handlectx->mutex);

    ret = OMX_SendCommand(s->handle, OMX_CommandPortEnable, s->out_port, NULL);
    if (ret != OMX_ErrorNone) {
        av_log(s->avctx, AV_LOG_ERROR, "OMX_CommandPortEnable %d Error\n", __LINE__);
        return AVERROR_UNKNOWN;
    }

    if (out_port_params.format.video.eColorFormat != OMX_COLOR_FormatUnused)
        avctx->pix_fmt = ff_omx_get_pix_fmt(out_port_params.format.video.eColorFormat);
    if (out_port_params.format.video.nFrameWidth &&
        out_port_params.format.video.nFrameHeight)
        ff_set_dimensions(avctx,
                          out_port_params.format.video.nFrameWidth,
                          out_port_params.format.video.nFrameHeight);

    ff_omx_wait_for_port_state(s, OMX_DirOutput, 1);

    for (i = 0; i < s->num_out_buffers && err == OMX_ErrorNone; i++)
        err = OMX_FillThisBuffer(s->handle, s->out_buffer_headers[i]);
    if (err != OMX_ErrorNone) {
        for (; i < s->num_out_buffers; i++)
            s->done_out_buffers[s->num_done_out_buffers++] = s->out_buffer_headers[i];
    }

    return 0;
}

static av_cold int omx_component_init(AVCodecContext *avctx, const char *role)
{
    OMXCodecContext *s = avctx->priv_data;
    OMX_VIDEO_PARAM_PORTFORMATTYPE video_port_format = { 0 };
    OMX_PARAM_PORTDEFINITIONTYPE in_port_params = { 0 }, out_port_params = { 0 };
    OMX_ERRORTYPE err;
    OMX_INDEXTYPE index;
    int i;

    int ret = ff_omx_component_init(avctx, role, &in_port_params, &out_port_params);

    if (ret < 0)
        return ret;

    in_port_params.bEnabled   = OMX_TRUE;
    in_port_params.bPopulated = OMX_FALSE;
    in_port_params.eDomain    = OMX_PortDomainVideo;
    //in_port_params.format.video.eCompressionFormat = s->compressFormat;
    in_port_params.format.video.pNativeRender         = NULL;
    in_port_params.format.video.bFlagErrorConcealment = OMX_FALSE;
    in_port_params.format.video.nFrameWidth  = 0;
    in_port_params.format.video.nFrameHeight = 0;

    //in_port_params.nBufferCountActual = in_port_params.nBufferCountActual + 10;

    err = OMX_SetParameter(s->handle, OMX_IndexParamPortDefinition, &in_port_params);
    CHECK(err);
    err = OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition, &in_port_params);
    CHECK(err);
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
    out_port_params.format.video.eColorFormat  = OMX_COLOR_FormatYUV420SemiPlanar;
    out_port_params.format.video.eCompressionFormat = OMX_VIDEO_CodingUnused;

    out_port_params.nBufferCountActual = out_port_params.nBufferCountActual + 10;

    err = OMX_SetParameter(s->handle, OMX_IndexParamPortDefinition, &out_port_params);
    CHECK(err);
    err = OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition, &out_port_params);
    CHECK(err);
    s->num_out_buffers = out_port_params.nBufferCountActual;

    INIT_STRUCT(video_port_format);
    video_port_format.nPortIndex = s->out_port;
    video_port_format.eColorFormat = OMX_COLOR_FormatYUV420SemiPlanar;
    video_port_format.eCompressionFormat = OMX_VIDEO_CodingUnused;
    err = OMX_SetParameter(s->handle, OMX_IndexParamVideoPortFormat, &video_port_format);
    CHECK(err);

    err = OMX_SendCommand(s->handle, OMX_CommandPortDisable, s->out_port, NULL);
    CHECK(err);

    if ((ret = ff_omx_wait_for_port_state(s, OMX_DirOutput, 0)) < 0) {
        av_log(avctx, AV_LOG_ERROR, "Can't wait %d port disable\n", s->out_port);
        return ret;
    }

    if (s->scale_width > 0 && s->scale_height > 0) {
        if (OMX_GetExtensionIndex(s->handle,
                                  (OMX_STRING)"OMX.realtek.android.index.notify_ve_scaling",
                                  &index) == OMX_ErrorNone) {
            OMX_U32 param[4];
            param[0] = s->scale_width;
            param[1] = s->scale_height;
            param[2] = 0; // fps
            param[3] = 0; // auto_resize
            err = OMX_SetParameter(s->handle, index, &param);
            CHECK(err);
        }
    }

    if(s->deinterlace != -1) {
        if (OMX_GetExtensionIndex(s->handle,
                                  (OMX_STRING)"OMX.realtek.android.index.videoDeInterlaced",
                                  &index) == OMX_ErrorNone) {
            OMX_U32 param[1];
            // 0: auto (default) / 1: on / 2: off
            param[0] = s->deinterlace ? 1 : 2;
            OMX_SetParameter(s->handle, index, &param);
        }
    }

#if 0
    if(s->search_I_frm == 1) {
        if (OMX_GetExtensionIndex(s->handle,
                                  (OMX_STRING)"OMX.realtek.android.index.setSearchIFrm",
                                  &index) == OMX_ErrorNone) {
            OMX_U32 param[1];
            param[0] = s->search_I_err_tolerance;
            if(s->search_I_err_tolerance>=0 && s->search_I_err_tolerance<=100)
                OMX_SetParameter(s->handle, index, &param);
        }
    }
#endif

    err = OMX_SendCommand(s->handle, OMX_CommandStateSet, OMX_StateIdle, NULL);
    CHECK(err);

    s->in_buffer_headers  = av_mallocz(sizeof(OMX_BUFFERHEADERTYPE*) * s->num_in_buffers);
    s->free_in_buffers    = av_mallocz(sizeof(OMX_BUFFERHEADERTYPE*) * s->num_in_buffers);
    if (!s->in_buffer_headers || !s->free_in_buffers)
        return AVERROR(ENOMEM);
    for (i = 0; i < s->num_in_buffers && err == OMX_ErrorNone; i++) {
        if (s->input_zerocopy)
            err = OMX_UseBuffer(s->handle, &s->in_buffer_headers[i], s->in_port, s, in_port_params.nBufferSize, NULL);
        else
            err = OMX_AllocateBuffer(s->handle, &s->in_buffer_headers[i],  s->in_port, s, in_port_params.nBufferSize);
        if (err == OMX_ErrorNone)
            s->in_buffer_headers[i]->pAppPrivate = s->in_buffer_headers[i]->pOutputPortPrivate = NULL;
    }
    CHECK(err);
    s->num_in_buffers = i;

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

    for (i = 0; i < s->num_in_buffers; i++)
        s->free_in_buffers[s->num_free_in_buffers++] = s->in_buffer_headers[i];

    return omx_reconfigure_output_buffer(avctx);
}

/*
static av_cold int omx_send_extradata(AVCodecContext *avctx)
{
    AVBufferRef *ref = NULL;
    OMX_BUFFERHEADERTYPE* buffer = ff_omx_get_buffer(&s->input_mutex, &s->input_cond,
                                                     &s->num_free_in_buffers, s->free_in_buffers, 1);

    if (!buffer) {
        av_log(avctx, AV_LOG_ERROR, "Failed to get input buffer\n");
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    if (s->input_zerocopy) {
        if ((ret = av_packet_make_refcounted(&pkt)) < 0)
            goto loopfail;

        av_assert0(pkt.buf);

        if (!(ref = av_buffer_ref(pkt.buf))) {
            ret = AVERROR(ENOMEM);
            goto loopfail;
        }

        buffer->pBuffer     = ref->data;
        buffer->nFilledLen  = ref->size;
        buffer->pAppPrivate = ref;

        pkt.size = 0;
    } else {
        int size = FFMIN(pkt.size, buffer->nAllocLen);
        av_assert0(size >= 0);
        av_assert0(buffer->pBuffer);
        memcpy(buffer->pBuffer, pkt.data, size);
        buffer->nFilledLen = size;
        pkt.data += size;
        pkt.size -= size;
    }

    buffer->nFlags = 0;
    if (eos)
        buffer->nFlags = OMX_BUFFERFLAG_EOS;
    else if (!pkt.size)
        buffer->nFlags = OMX_BUFFERFLAG_ENDOFFRAME;

    buffer->nOffset = 0;
    buffer->nTimeStamp = to_omx_ticks(av_rescale_q(pkt.pts, avctx->time_base, AV_TIME_BASE_Q));

    if ((err = OMX_EmptyThisBuffer(s->handle, buffer)) != OMX_ErrorNone) {
        av_log(avctx, AV_LOG_ERROR, "OMX_EmptyThisBuffer failed: %x\n", err);
        ret = AVERROR_UNKNOWN;
    }
}
*/

static av_cold int omx_decode_init(AVCodecContext *avctx)
{
    const char *role;

    int ret = ff_omx_codec_init(avctx);
    if (ret < 0)
        return ret;

    switch (avctx->codec->id) {
    case AV_CODEC_ID_MPEG4:
        role = "video_decoder.mpeg4";
        break;
    case AV_CODEC_ID_H264:
        avctx->ticks_per_frame = 2;
        role = "video_decoder.avc";
        break;
    case AV_CODEC_ID_HEVC:
        role = "video_decoder.hevc";
        break;
    case AV_CODEC_ID_VP9:
        role = "video_decoder.vp9";
        break;
    default:
        return AVERROR(ENOSYS);
    }

    if ((ret = omx_component_init(avctx, role)) < 0)
        goto fail;

fail:
    return ret;
}

static void release_buffer(void *opaque, uint8_t *data)
{
    struct OMXOutputContext *bufctx = opaque;
    OMXHandleContext* hctx = (void*)bufctx->href->data;
    OMX_ERRORTYPE err;

    int reuse = 0;

    pthread_mutex_lock(&hctx->mutex);

    if (hctx->ctx && hctx->ctx->output_reconfig_count == bufctx->output_reconfig_count) {
        reuse = 1;
        pthread_mutex_unlock(&hctx->mutex);
        err = OMX_FillThisBuffer(hctx->handle, bufctx->buffer);
        pthread_mutex_lock(&hctx->mutex);
        if (err != OMX_ErrorNone) {
            av_log(hctx->ctx->avctx, AV_LOG_ERROR, "OMX_FillThisBuffer failed: %x\n", err);
            if (hctx->ctx)
                ff_omx_append_buffer(hctx->ctx, &hctx->ctx->num_done_out_buffers, hctx->ctx->done_out_buffers, bufctx->buffer);
            else
                reuse = 0;
        }
    }
    pthread_mutex_unlock(&hctx->mutex);

    if (!reuse)
        OMX_FreeBuffer(hctx->handle, bufctx->buffer->nOutputPortIndex, bufctx->buffer);

    av_buffer_unref(&bufctx->href);
    av_freep(&bufctx);
}

static int omx_receive_frame(AVCodecContext *avctx, AVFrame* frame)
{
    OMXCodecContext *s = avctx->priv_data;
    struct OMXOutputContext *bufctx = NULL;
    OMX_BUFFERHEADERTYPE *buffer;
    OMX_ERRORTYPE err;
    int ret = 0;

    buffer = ff_omx_get_buffer(s, &s->num_done_out_buffers, s->done_out_buffers, s->eos_sent);

    if (!buffer)
        return AVERROR(EAGAIN);

    if (buffer->nFlags & OMX_BUFFERFLAG_EOS) {
        s->got_eos = 1;
        if (!buffer->nFilledLen) {
            ret = AVERROR_EOF;
            goto end;
        }
    }

    if (!buffer->nFilledLen)
        return AVERROR(EINVAL);

    if ((ret = ff_decode_frame_props(avctx, frame)) < 0)
        goto end;

    frame->width = avctx->width;
    frame->height = avctx->height;

    if (avctx->pkt_timebase.num && avctx->pkt_timebase.den)
        frame->pts = av_rescale_q(from_omx_ticks(buffer->nTimeStamp), AV_TIME_BASE_Q, avctx->pkt_timebase);
    else
        frame->pts = from_omx_ticks(buffer->nTimeStamp);

    frame->pkt_dts = AV_NOPTS_VALUE;

    bufctx = av_mallocz(sizeof(*bufctx));
    if (!bufctx) {
        ret = AVERROR(ENOMEM);
        goto end;
    }

    bufctx->href = av_buffer_ref(s->ref);
    if (!bufctx->href) {
        ret = AVERROR(ENOMEM);
        goto end;
    }

    bufctx->buffer = buffer;
    bufctx->output_reconfig_count = s->output_reconfig_count;

    frame->buf[0] = av_buffer_create(buffer->pBuffer, buffer->nFilledLen, release_buffer, bufctx, AV_BUFFER_FLAG_READONLY);
    if (!frame->buf[0]) {
        ret = AVERROR(ENOMEM);
        goto end;
    }

    frame->data[0] = buffer->pBuffer;
    frame->linesize[0] = s->stride;
    frame->data[1] = frame->data[0] + s->stride * s->plane_size;
    if (avctx->pix_fmt == AV_PIX_FMT_NV12) {
        frame->linesize[1] = s->stride;
    } else {
        // FIXME: assuming chroma plane's stride is 1/2 of luma plane's for YV12
        frame->linesize[1] = frame->linesize[2] = s->stride / 2;
        frame->data[2] = frame->data[1] + s->stride * s->plane_size / 4;
    }

    frame->data[3] = (void*)buffer;

end:
    if (ret < 0) {
        if (bufctx) {
            av_buffer_unref(&bufctx->href);
            av_free(bufctx);
        }

        err = OMX_FillThisBuffer(s->handle, buffer);
        if (err != OMX_ErrorNone) {
            ff_omx_append_buffer(s, &s->num_done_out_buffers, s->done_out_buffers, buffer);
            av_log(avctx, AV_LOG_ERROR, "OMX_FillThisBuffer failed: %x\n", err);
            ret = AVERROR_UNKNOWN;
        }
    }

    return ret;
}

static int omx_send_packet(AVCodecContext *avctx)
{
    OMXCodecContext *s = avctx->priv_data;
    AVPacket pkt = {0};
    int ret = ff_decode_get_packet(avctx, &pkt);
    OMX_ERRORTYPE err;
    int eos;

    if (ret < 0 && ret != AVERROR_EOF) {
        if (ret != AVERROR(EAGAIN))
            av_log(avctx, AV_LOG_ERROR, "Failed to get packet\n");
        return ret;
    }

    ret = 0;

    eos = pkt.size == 0;

    do {
        AVBufferRef *ref = NULL;
        OMX_BUFFERHEADERTYPE* buffer = ff_omx_get_buffer(s, &s->num_free_in_buffers, s->free_in_buffers, 1);

        if (!buffer) {
            av_log(avctx, AV_LOG_ERROR, "Failed to get input buffer\n");
            ret = AVERROR_UNKNOWN;
            goto fail;
        }

        if (s->input_zerocopy) {
            if ((ret = av_packet_make_refcounted(&pkt)) < 0)
                goto loopfail;

            av_assert0(pkt.buf);

            if (!(ref = av_buffer_ref(pkt.buf))) {
                ret = AVERROR(ENOMEM);
                goto loopfail;
            }

            buffer->pBuffer     = ref->data;
            buffer->nFilledLen  = ref->size;
            buffer->pAppPrivate = ref;

            pkt.size = 0;
        } else {
            int size = FFMIN(pkt.size, buffer->nAllocLen);
            av_assert0(size >= 0);
            av_assert0(buffer->pBuffer);
            memcpy(buffer->pBuffer, pkt.data, size);
            buffer->nFilledLen = size;
            pkt.data += size;
            pkt.size -= size;
        }

        buffer->nFlags = 0;
        if (eos)
            buffer->nFlags = OMX_BUFFERFLAG_EOS;
        else if (!pkt.size)
            buffer->nFlags = OMX_BUFFERFLAG_ENDOFFRAME;

        buffer->nOffset = 0;
        buffer->nTimeStamp = to_omx_ticks(av_rescale_q(pkt.pts, avctx->pkt_timebase, AV_TIME_BASE_Q));

        if ((err = OMX_EmptyThisBuffer(s->handle, buffer)) != OMX_ErrorNone) {
            av_log(avctx, AV_LOG_ERROR, "OMX_EmptyThisBuffer failed: %x\n", err);
            ret = AVERROR_UNKNOWN;
        }

loopfail:
        if (ret < 0) {
            // Put the buffer back in the queue
            ff_omx_append_buffer(s, &s->num_free_in_buffers, s->free_in_buffers, buffer);
            if (ref)
                av_buffer_unref(&ref);
            goto fail;
        }
    } while (pkt.size);

fail:
    av_packet_unref(&pkt);
    return ret;
}

static int omx_decode_receive_frame(AVCodecContext *avctx, AVFrame *frame)
{
    OMXCodecContext *s = avctx->priv_data;
    int ret = 0;
    int output_pending_reconfigure = 0;
    int num_done_out_buffers = 0;

    if (s->got_eos)
        return AVERROR_EOF;

retry:
    if ((ret = omx_receive_frame(avctx, frame)) != AVERROR(EAGAIN))
        return ret;

    pthread_mutex_lock(&s->handlectx->mutex);
    while (!s->output_pending_reconfigure && !s->num_free_in_buffers && !s->num_done_out_buffers)
        pthread_cond_wait(&s->handlectx->cond, &s->handlectx->mutex);
    output_pending_reconfigure = s->output_pending_reconfigure;
    num_done_out_buffers = s->num_done_out_buffers;
    pthread_mutex_unlock(&s->handlectx->mutex);

    // Wait until we've drained all our frames to do the reconfigure or take input.
    if (num_done_out_buffers)
        goto retry;

    if (output_pending_reconfigure) {
        ret = omx_reconfigure_output_buffer(avctx);
        if (ret < 0)
            return ret;
        goto retry;
    }

    av_assert0(!s->eos_sent);

    ret = omx_send_packet(avctx);
    if (ret >= 0)
        ret = AVERROR(EAGAIN);
    return ret;
}

static av_cold int omx_decode_end(AVCodecContext *avctx)
{
    OMXCodecContext *s = avctx->priv_data;

    ff_omx_cleanup(s);
    return 0;
}

static void omx_decode_flush(AVCodecContext *avctx)
{
    OMXCodecContext *s = avctx->priv_data;

    s->eos_sent = s->got_eos = OMX_FALSE;

    pthread_mutex_lock(&s->handlectx->mutex);
    s->flush_in_port_done = OMX_FALSE;
    s->flush_out_port_done = OMX_FALSE;
    pthread_mutex_unlock(&s->handlectx->mutex);

    OMX_SendCommand(s->handle, OMX_CommandFlush, OMX_ALL, NULL);
    ff_omx_wait_for_port_flush(s, OMX_DirOutput);
    ff_omx_wait_for_port_flush(s, OMX_DirInput);
}

#define OFFSET(x) offsetof(OMXCodecContext, x)
#define VDE AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM | AV_OPT_FLAG_ENCODING_PARAM
#define VD  AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM

static const AVOption options[] = {
    { "omx_libname", "OpenMAX library name", OFFSET(libname), AV_OPT_TYPE_STRING, { 0 }, 0, 0, VDE },
    { "omx_libprefix", "OpenMAX library prefix", OFFSET(libprefix), AV_OPT_TYPE_STRING, { 0 }, 0, 0, VDE },
    { "zerocopy", "Try to avoid copying input packets if possible", OFFSET(input_zerocopy), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, VDE },
    { "hwdeint_mode", "Deinterlace video (-1 = auto)", OFFSET(deinterlace), AV_OPT_TYPE_BOOL, { .i64 = -1 }, -1, 1, VDE },
    { "output_size", "Scale output", OFFSET(scale_width), AV_OPT_TYPE_IMAGE_SIZE, { .i64 = 0 }, 0, INT_MAX, VDE },
    { NULL }
};

#define OMXDEC(namev, longname, idv, bsf) \
static const AVClass omx_##namev##dec_class = { \
    .class_name = #namev "_omx", \
    .item_name  = av_default_item_name, \
    .option     = options, \
    .version    = LIBAVUTIL_VERSION_INT, \
}; \
AVCodec ff_##namev##_omx_decoder = { \
    .name             = #namev "_omx", \
    .long_name        = NULL_IF_CONFIG_SMALL("OpenMAX IL " longname " video decoder"), \
    .type             = AVMEDIA_TYPE_VIDEO, \
    .id               = idv, \
    .priv_data_size   = sizeof(OMXCodecContext), \
    .capabilities     = AV_CODEC_CAP_DELAY, \
    .caps_internal    = FF_CODEC_CAP_INIT_THREADSAFE | FF_CODEC_CAP_INIT_CLEANUP | FF_CODEC_CAP_SETS_PKT_DTS, \
    .priv_class       = &omx_##namev##dec_class, \
    .init             = omx_decode_init, \
    .receive_frame    = omx_decode_receive_frame, \
    .close            = omx_decode_end, \
    .flush            = omx_decode_flush, \
    .bsfs             = bsf, \
};

OMXDEC(mpeg4, "MPEG-4", AV_CODEC_ID_MPEG4, NULL)
OMXDEC(h264, "H.264", AV_CODEC_ID_H264, "h264_mp4toannexb")
OMXDEC(hevc, "HEVC", AV_CODEC_ID_HEVC, "hevc_mp4toannexb")
OMXDEC(vp9, "VP9", AV_CODEC_ID_VP9, NULL)
