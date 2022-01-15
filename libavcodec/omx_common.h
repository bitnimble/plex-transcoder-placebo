/*
 * OMX common utilities
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

#ifndef AVCODEC_OMX_COMMON_H
#define AVCODEC_OMX_COMMON_H

#include "config.h"

#if CONFIG_OMX_RPI
#define OMX_SKIP64BIT
#endif

#include <dlfcn.h>
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
#include "internal.h"

#ifdef OMX_SKIP64BIT
static inline OMX_TICKS to_omx_ticks(int64_t value)
{
    OMX_TICKS s;
    s.nLowPart  = value & 0xffffffff;
    s.nHighPart = value >> 32;
    return s;
}
static inline int64_t from_omx_ticks(OMX_TICKS value)
{
    return (((int64_t)value.nHighPart) << 32) | value.nLowPart;
}
#else
#define to_omx_ticks(x) (x)
#define from_omx_ticks(x) (x)
#endif

#define INIT_STRUCT(x) do {                                               \
        x.nSize = sizeof(x);                                              \
        x.nVersion = s->version;                                          \
    } while (0)
#define CHECK(x) do {                                                     \
        if (x != OMX_ErrorNone) {                                         \
            av_log(avctx, AV_LOG_ERROR,                                   \
                   "err %x (%d) on line %d\n", x, x, __LINE__);           \
            return AVERROR_UNKNOWN;                                       \
        }                                                                 \
    } while (0)

typedef struct OMXContext {
    void *lib;
    void *lib2;
    OMX_ERRORTYPE (*ptr_Init)(void);
    OMX_ERRORTYPE (*ptr_Deinit)(void);
    OMX_ERRORTYPE (*ptr_ComponentNameEnum)(OMX_STRING, OMX_U32, OMX_U32);
    OMX_ERRORTYPE (*ptr_GetHandle)(OMX_HANDLETYPE*, OMX_STRING, OMX_PTR, OMX_CALLBACKTYPE*);
    OMX_ERRORTYPE (*ptr_FreeHandle)(OMX_HANDLETYPE);
    OMX_ERRORTYPE (*ptr_GetComponentsOfRole)(OMX_STRING, OMX_U32*, OMX_U8**);
    OMX_ERRORTYPE (*ptr_GetRolesOfComponent)(OMX_STRING, OMX_U32*, OMX_U8**);
    void (*host_init)(void);
} OMXContext;

av_cold OMXContext *ff_omx_init(void *logctx, const char *libname, const char *prefix);

typedef struct OMXHandleContext {
    struct OMXCodecContext *ctx;
    OMX_HANDLETYPE handle;
    OMXContext *omx_context;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int terminating;
} OMXHandleContext;

typedef OMX_ERRORTYPE (*OMXRTKCopyFunction)(
        OMX_IN OMX_HANDLETYPE hComponent,
        OMX_IN OMX_BUFFERHEADERTYPE* dst_buf,
        OMX_IN OMX_BUFFERHEADERTYPE* src_buf,
        OMX_IN OMX_U32 size);

typedef struct OMXCodecContext {
    const AVClass *class;
    char *libname;
    char *libprefix;

    AVCodecContext *avctx;

    char component_name[OMX_MAX_STRINGNAME_SIZE];
    OMX_VERSIONTYPE version;
    OMX_HANDLETYPE handle;
    int in_port, out_port;
    OMX_COLOR_FORMATTYPE color_format;
    int stride, plane_size;

    int num_in_buffers, num_out_buffers;
    OMX_BUFFERHEADERTYPE **in_buffer_headers;
    OMX_BUFFERHEADERTYPE **out_buffer_headers;
    int num_free_in_buffers;
    OMX_BUFFERHEADERTYPE **free_in_buffers;
    int num_done_out_buffers;
    OMX_BUFFERHEADERTYPE **done_out_buffers;

    OMX_STATETYPE state;
    OMX_ERRORTYPE error;

    int flush_in_port_done, flush_out_port_done;
    int in_port_state, out_port_state;
    int output_pending_reconfigure;
    uint64_t output_reconfig_count;

    int eos_sent, got_eos;

    uint8_t *output_buf;
    int output_buf_size;

    int input_zerocopy;
    int profile;

    OMXHandleContext *handlectx;
    AVBufferRef *ref;

    OMXRTKCopyFunction copy_func;

    int deinterlace;
    int scale_width, scale_height;
} OMXCodecContext;

#define NB_MUTEX_CONDS 6
#define OFF(field) offsetof(OMXCodecContext, field)
DEFINE_OFFSET_ARRAY(OMXCodecContext, omx_codec_context, mutex_cond_inited_cnt,
                    (OFF(input_mutex), OFF(output_mutex), OFF(state_mutex)),
                    (OFF(input_cond),  OFF(output_cond),  OFF(state_cond)));

av_cold int ff_omx_codec_init(AVCodecContext *avctx);

void ff_omx_append_buffer_locked(OMXCodecContext *s,
                                 int* array_size, OMX_BUFFERHEADERTYPE **array,
                                 OMX_BUFFERHEADERTYPE *buffer);

void ff_omx_append_buffer(OMXCodecContext *s,
                          int* array_size, OMX_BUFFERHEADERTYPE **array,
                          OMX_BUFFERHEADERTYPE *buffer);

OMX_BUFFERHEADERTYPE *ff_omx_get_buffer_locked(OMXCodecContext *s,
                                               int* array_size, OMX_BUFFERHEADERTYPE **array,
                                               int wait);

OMX_BUFFERHEADERTYPE *ff_omx_get_buffer(OMXCodecContext *s,
                                        int* array_size, OMX_BUFFERHEADERTYPE **array,
                                        int wait);

av_cold int ff_omx_component_init(AVCodecContext *avctx, const char *role,
                                  OMX_PARAM_PORTDEFINITIONTYPE* in,
                                  OMX_PARAM_PORTDEFINITIONTYPE* out);

av_cold int ff_omx_wait_for_state(OMXCodecContext *s, OMX_STATETYPE state);
av_cold int ff_omx_wait_for_port_flush(OMXCodecContext *s, enum OMX_DIRTYPE dir);
av_cold int ff_omx_wait_for_port_state(OMXCodecContext *s, enum OMX_DIRTYPE dir, int enabled);

av_cold void ff_omx_cleanup(OMXCodecContext *s);

enum AVPixelFormat ff_omx_get_pix_fmt(enum OMX_COLOR_FORMATTYPE omxfmt);
enum OMX_COLOR_FORMATTYPE ff_omx_get_color_format(enum AVPixelFormat pixfmt);

struct OMXOutputContext {
    AVBufferRef *href;
    OMX_BUFFERHEADERTYPE *buffer;
    uint64_t output_reconfig_count;
};

#endif /* AVCODEC_OMX_COMMON_H */
