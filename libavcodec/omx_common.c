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

#include "omx_common.h"

#include "libavutil/avassert.h"

static av_cold void *dlsym_prefixed(void *handle, const char *symbol, const char *prefix)
{
    char buf[50];
    snprintf(buf, sizeof(buf), "%s%s", prefix ? prefix : "", symbol);
    return dlsym(handle, buf);
}

static av_cold int omx_try_load(OMXContext *s, void *logctx,
                                const char *libname, const char *prefix,
                                const char *libname2)
{
    if (libname2) {
        s->lib2 = dlopen(libname2, RTLD_NOW | RTLD_GLOBAL);
        if (!s->lib2) {
            av_log(logctx, AV_LOG_WARNING, "%s not found\n", libname2);
            return AVERROR_ENCODER_NOT_FOUND;
        }
        s->host_init = dlsym(s->lib2, "bcm_host_init");
        if (!s->host_init) {
            av_log(logctx, AV_LOG_WARNING, "bcm_host_init not found\n");
            dlclose(s->lib2);
            s->lib2 = NULL;
            return AVERROR_ENCODER_NOT_FOUND;
        }
    }
    s->lib = dlopen(libname, RTLD_NOW | RTLD_GLOBAL);
    if (!s->lib) {
        av_log(logctx, AV_LOG_WARNING, "%s not found\n", libname);
        return AVERROR_ENCODER_NOT_FOUND;
    }
    s->ptr_Init                = dlsym_prefixed(s->lib, "OMX_Init", prefix);
    s->ptr_Deinit              = dlsym_prefixed(s->lib, "OMX_Deinit", prefix);
    s->ptr_ComponentNameEnum   = dlsym_prefixed(s->lib, "OMX_ComponentNameEnum", prefix);
    s->ptr_GetHandle           = dlsym_prefixed(s->lib, "OMX_GetHandle", prefix);
    s->ptr_FreeHandle          = dlsym_prefixed(s->lib, "OMX_FreeHandle", prefix);
    s->ptr_GetComponentsOfRole = dlsym_prefixed(s->lib, "OMX_GetComponentsOfRole", prefix);
    s->ptr_GetRolesOfComponent = dlsym_prefixed(s->lib, "OMX_GetRolesOfComponent", prefix);
    if (!s->ptr_Init || !s->ptr_Deinit || !s->ptr_ComponentNameEnum ||
        !s->ptr_GetHandle || !s->ptr_FreeHandle ||
        !s->ptr_GetComponentsOfRole || !s->ptr_GetRolesOfComponent) {
        av_log(logctx, AV_LOG_WARNING, "Not all functions found in %s\n", libname);
        dlclose(s->lib);
        s->lib = NULL;
        if (s->lib2)
            dlclose(s->lib2);
        s->lib2 = NULL;
        return AVERROR_ENCODER_NOT_FOUND;
    }
    return 0;
}

OMXContext *ff_omx_init(void *logctx, const char *libname, const char *prefix)
{
    static const char * const libnames[] = {
#if CONFIG_OMX_RPI
        "/opt/vc/lib/libopenmaxil.so", "/opt/vc/lib/libbcm_host.so",
#else
        "libOMX_Core.so", NULL,
        "libOmxCore.so", NULL,
#endif
        NULL
    };
    const char* const* nameptr;
    int ret = AVERROR_ENCODER_NOT_FOUND;
    OMXContext *omx_context;

    omx_context = av_mallocz(sizeof(*omx_context));
    if (!omx_context)
        return NULL;
    if (libname) {
        ret = omx_try_load(omx_context, logctx, libname, prefix, NULL);
        if (ret < 0) {
            av_free(omx_context);
            return NULL;
        }
    } else {
        for (nameptr = libnames; *nameptr; nameptr += 2)
            if (!(ret = omx_try_load(omx_context, logctx, nameptr[0], prefix, nameptr[1])))
                break;
        if (!*nameptr) {
            av_free(omx_context);
            return NULL;
        }
    }

    if (omx_context->host_init)
        omx_context->host_init();
    omx_context->ptr_Init();
    return omx_context;
}

static av_cold void free_handle_ctx(void *opaque, uint8_t *data)
{
    OMXHandleContext *s = (void*)data;
    if (s->handle)
        s->omx_context->ptr_FreeHandle(s->handle);
    if (s->omx_context) {
        s->omx_context->ptr_Deinit();
        dlclose(s->omx_context->lib);
        av_free(s->omx_context);
    }
    pthread_mutex_destroy(&s->mutex);
    pthread_cond_destroy(&s->cond);
    av_free(s);
}

int ff_omx_codec_init(AVCodecContext *avctx)
{
    OMXCodecContext *s = avctx->priv_data;

    if (!(s->handlectx = av_mallocz(sizeof(OMXHandleContext))))
        return AVERROR(ENOMEM);
    s->handlectx->ctx = s;
    pthread_mutex_init(&s->handlectx->mutex, NULL);
    pthread_cond_init(&s->handlectx->cond, NULL);
    if (!(s->ref = av_buffer_create((void*)s->handlectx, sizeof(*s->handlectx), free_handle_ctx, NULL, 0))) {
        pthread_mutex_destroy(&s->handlectx->mutex);
        pthread_cond_destroy(&s->handlectx->cond);
        av_free(s->handlectx);
        return AVERROR(ENOMEM);
    }

    s->handlectx->omx_context = ff_omx_init(avctx, s->libname, s->libprefix);
    if (!s->handlectx->omx_context)
        return AVERROR_ENCODER_NOT_FOUND;

    s->avctx = avctx;
    s->state = OMX_StateLoaded;
    s->error = OMX_ErrorNone;
    s->in_port_state = s->out_port_state = 1;

    return 0;
}

void ff_omx_append_buffer_locked(OMXCodecContext *s,
                                 int* array_size, OMX_BUFFERHEADERTYPE **array,
                                 OMX_BUFFERHEADERTYPE *buffer)
{
    array[(*array_size)++] = buffer;
    pthread_cond_broadcast(&s->handlectx->cond);
}

void ff_omx_append_buffer(OMXCodecContext *s,
                          int* array_size, OMX_BUFFERHEADERTYPE **array,
                          OMX_BUFFERHEADERTYPE *buffer)
{
    pthread_mutex_lock(&s->handlectx->mutex);
    ff_omx_append_buffer_locked(s, array_size, array, buffer);
    pthread_mutex_unlock(&s->handlectx->mutex);
}

OMX_BUFFERHEADERTYPE *ff_omx_get_buffer_locked(OMXCodecContext *s,
                                               int* array_size, OMX_BUFFERHEADERTYPE **array,
                                               int wait)
{
    OMX_BUFFERHEADERTYPE *buffer;
    if (wait) {
        while (!*array_size)
           pthread_cond_wait(&s->handlectx->cond, &s->handlectx->mutex);
    }
    if (*array_size > 0) {
        buffer = array[0];
        (*array_size)--;
        memmove(&array[0], &array[1], (*array_size) * sizeof(OMX_BUFFERHEADERTYPE*));
    } else {
        buffer = NULL;
    }
    return buffer;
}

OMX_BUFFERHEADERTYPE *ff_omx_get_buffer(OMXCodecContext *s,
                                        int* array_size, OMX_BUFFERHEADERTYPE **array,
                                        int wait)
{
    OMX_BUFFERHEADERTYPE *buffer;
    pthread_mutex_lock(&s->handlectx->mutex);
    buffer = ff_omx_get_buffer_locked(s, array_size, array, wait);
    pthread_mutex_unlock(&s->handlectx->mutex);
    return buffer;
}

static OMX_ERRORTYPE event_handler(OMX_HANDLETYPE component, OMX_PTR app_data, OMX_EVENTTYPE event,
                                   OMX_U32 data1, OMX_U32 data2, OMX_PTR event_data)
{
    OMXHandleContext *hctx = app_data;
    OMXCodecContext *s = hctx->ctx;
    if (!s)
        return OMX_ErrorNone;
    // This uses casts in the printfs, since OMX_U32 actually is a typedef for
    // unsigned long in official header versions (but there are also modified
    // versions where it is something else).
    switch (event) {
    case OMX_EventError:
        pthread_mutex_lock(&hctx->mutex);
        av_log(s->avctx, AV_LOG_ERROR, "OMX error %"PRIx32"\n", (uint32_t) data1);
        s->error = data1;
        pthread_cond_broadcast(&hctx->cond);
        pthread_mutex_unlock(&hctx->mutex);
        break;
    case OMX_EventCmdComplete:
        if (data1 == OMX_CommandStateSet) {
            pthread_mutex_lock(&hctx->mutex);
            s->state = data2;
            av_log(s->avctx, AV_LOG_VERBOSE, "OMX state changed to %"PRIu32"\n", (uint32_t) data2);
            pthread_cond_broadcast(&hctx->cond);
            pthread_mutex_unlock(&hctx->mutex);
        } else if (data1 == OMX_CommandPortEnable || data1 == OMX_CommandPortDisable) {
            int state = (data1 == OMX_CommandPortEnable);
            av_log(s->avctx, AV_LOG_VERBOSE, "OMX port %"PRIu32" %s\n", (uint32_t) data2,
                   state ? "enabled" : "disabled");
            pthread_mutex_lock(&hctx->mutex);
            if (data2 == s->in_port)
                s->in_port_state = state;
            else if (data2 == s->out_port)
                s->out_port_state = state;
            pthread_cond_broadcast(&hctx->cond);
            pthread_mutex_unlock(&hctx->mutex);
        }  else if (data1 == OMX_CommandFlush) {
            pthread_mutex_lock(&hctx->mutex);
            if(data2 == s->in_port)
                s->flush_in_port_done = 1;
            else if(data2 == s->out_port)
                s->flush_out_port_done = 1;
            pthread_cond_broadcast(&hctx->cond);
            pthread_mutex_unlock(&hctx->mutex);
        } else {
            av_log(s->avctx, AV_LOG_VERBOSE, "OMX command complete, command %"PRIu32", value %"PRIu32"\n",
                                             (uint32_t) data1, (uint32_t) data2);
        }
        break;
    case OMX_EventPortSettingsChanged:
        if (data1 == s->out_port) {
            pthread_mutex_lock(&hctx->mutex);
            s->output_pending_reconfigure = 1;
            pthread_cond_broadcast(&hctx->cond);
            pthread_mutex_unlock(&hctx->mutex);
        }
        av_log(s->avctx, AV_LOG_VERBOSE, "OMX port %"PRIu32" settings changed\n", (uint32_t) data1);
        break;
    default:
        av_log(s->avctx, AV_LOG_VERBOSE, "OMX event %d %"PRIx32" %"PRIx32"\n",
                                         event, (uint32_t) data1, (uint32_t) data2);
        break;
    }
    return OMX_ErrorNone;
}

static OMX_ERRORTYPE empty_buffer_done(OMX_HANDLETYPE component, OMX_PTR app_data,
                                       OMX_BUFFERHEADERTYPE *buffer)
{
    int reused = 0;
    OMXHandleContext *hctx = app_data;
    if (buffer->pAppPrivate) {
        AVBufferRef *buf = buffer->pAppPrivate;
        av_buffer_unref(&buf);
        buffer->pAppPrivate = NULL;
        buffer->pBuffer = NULL;
    }

    pthread_mutex_lock(&hctx->mutex);
    if (!hctx->terminating) {
        OMXCodecContext *s = hctx->ctx;
        ff_omx_append_buffer_locked(s, &s->num_free_in_buffers, s->free_in_buffers, buffer);
        reused = 1;
    }
    pthread_mutex_unlock(&hctx->mutex);

    if (!reused)
        OMX_FreeBuffer(hctx->handle, buffer->nInputPortIndex, buffer);

    return OMX_ErrorNone;
}

static OMX_ERRORTYPE fill_buffer_done(OMX_HANDLETYPE component, OMX_PTR app_data,
                                      OMX_BUFFERHEADERTYPE *buffer)
{
    int reused = 0;
    OMXHandleContext *hctx = app_data;

    pthread_mutex_lock(&hctx->mutex);
    if (!hctx->terminating) {
        OMXCodecContext *s = hctx->ctx;
        ff_omx_append_buffer_locked(s, &s->num_done_out_buffers, s->done_out_buffers, buffer);
        reused = 1;
    }
    pthread_mutex_unlock(&hctx->mutex);

    if (!reused)
        OMX_FreeBuffer(hctx->handle, buffer->nOutputPortIndex, buffer);

    return OMX_ErrorNone;
}

static const OMX_CALLBACKTYPE callbacks = {
    event_handler,
    empty_buffer_done,
    fill_buffer_done
};

static int find_component(OMXContext *omx_context, void *logctx,
                          const char *role, char *str, int str_size)
{
    OMX_U32 i, num = 0;
    char **components;
    int ret = 0;

#if CONFIG_OMX_RPI
    if (av_strstart(role, "video_encoder.", NULL)) {
        av_strlcpy(str, "OMX.broadcom.video_encode", str_size);
        return 0;
    }
#endif
    omx_context->ptr_GetComponentsOfRole((OMX_STRING) role, &num, NULL);
    if (!num) {
        av_log(logctx, AV_LOG_WARNING, "No component for role %s found\n", role);
        return AVERROR_ENCODER_NOT_FOUND;
    }
    components = av_calloc(num, sizeof(*components));
    if (!components)
        return AVERROR(ENOMEM);
    for (i = 0; i < num; i++) {
        components[i] = av_mallocz(OMX_MAX_STRINGNAME_SIZE);
        if (!components[i]) {
            ret = AVERROR(ENOMEM);
            goto end;
        }
    }
    omx_context->ptr_GetComponentsOfRole((OMX_STRING) role, &num, (OMX_U8**) components);
    av_strlcpy(str, components[0], str_size);
end:
    for (i = 0; i < num; i++)
        av_free(components[i]);
    av_free(components);
    return ret;
}

av_cold int ff_omx_component_init(AVCodecContext *avctx, const char *role,
                                  OMX_PARAM_PORTDEFINITIONTYPE* in,
                                  OMX_PARAM_PORTDEFINITIONTYPE* out)
{
    OMXCodecContext *s = avctx->priv_data;
    OMX_PARAM_COMPONENTROLETYPE role_params = { 0 };
    OMX_PORT_PARAM_TYPE video_port_params = { 0 };
    OMX_ERRORTYPE err;
    int i;

    int ret = find_component(s->handlectx->omx_context, avctx, role, s->component_name, sizeof(s->component_name));
    if (ret < 0)
        return ret;

    av_log(avctx, AV_LOG_INFO, "Using %s\n", s->component_name);

    s->version.s.nVersionMajor = 1;
    s->version.s.nVersionMinor = 1;
    s->version.s.nRevision     = 2;

    err = s->handlectx->omx_context->ptr_GetHandle(&s->handlectx->handle, s->component_name, s->handlectx, (OMX_CALLBACKTYPE*) &callbacks);
    if (err != OMX_ErrorNone) {
        av_log(avctx, AV_LOG_ERROR, "OMX_GetHandle(%s) failed: %x\n", s->component_name, err);
        return AVERROR_UNKNOWN;
    }

    s->handle = s->handlectx->handle;

    // This one crashes the mediaserver on qcom, if used over IOMX
    INIT_STRUCT(role_params);
    av_strlcpy(role_params.cRole, role, sizeof(role_params.cRole));
    // Intentionally ignore errors on this one
    OMX_SetParameter(s->handle, OMX_IndexParamStandardComponentRole, &role_params);

    INIT_STRUCT(video_port_params);
    err = OMX_GetParameter(s->handle, OMX_IndexParamVideoInit, &video_port_params);
    CHECK(err);

    s->in_port = s->out_port = -1;
    for (i = 0; i < video_port_params.nPorts; i++) {
        int port = video_port_params.nStartPortNumber + i;
        OMX_PARAM_PORTDEFINITIONTYPE port_params = { 0 };
        INIT_STRUCT(port_params);
        port_params.nPortIndex = port;
        err = OMX_GetParameter(s->handle, OMX_IndexParamPortDefinition, &port_params);
        if (err != OMX_ErrorNone) {
            av_log(avctx, AV_LOG_WARNING, "port %d error %x\n", port, err);
            break;
        }
        if (port_params.eDir == OMX_DirInput && s->in_port < 0) {
            *in = port_params;
            s->in_port = port;
        } else if (port_params.eDir == OMX_DirOutput && s->out_port < 0) {
            *out = port_params;
            s->out_port = port;
        }
    }
    if (s->in_port < 0 || s->out_port < 0) {
        av_log(avctx, AV_LOG_ERROR, "No in or out port found (in %d out %d)\n", s->in_port, s->out_port);
        return AVERROR_UNKNOWN;
    }

    return 0;
}

int ff_omx_wait_for_state(OMXCodecContext *s, OMX_STATETYPE state)
{
    int ret = 0;
    pthread_mutex_lock(&s->handlectx->mutex);
    while (s->state != state && s->error == OMX_ErrorNone)
        pthread_cond_wait(&s->handlectx->cond, &s->handlectx->mutex);
    if (s->error != OMX_ErrorNone)
        ret = AVERROR_ENCODER_NOT_FOUND;
    pthread_mutex_unlock(&s->handlectx->mutex);
    return ret;
}

int ff_omx_wait_for_port_flush(OMXCodecContext *s, enum OMX_DIRTYPE dir)
{
    int ret = 0;
    int* done = (dir == OMX_DirOutput) ? &s->flush_out_port_done : &s->flush_in_port_done;
    pthread_mutex_lock(&s->handlectx->mutex);
    while (!*done)
        pthread_cond_wait(&s->handlectx->cond, &s->handlectx->mutex);
    pthread_mutex_unlock(&s->handlectx->mutex);
    return ret;
}

int ff_omx_wait_for_port_state(OMXCodecContext *s, enum OMX_DIRTYPE dir, int target)
{
    int ret = 0;
    int* state = (dir == OMX_DirOutput) ? &s->out_port_state : &s->in_port_state;
    pthread_mutex_lock(&s->handlectx->mutex);
    while (*state != target)
        pthread_cond_wait(&s->handlectx->cond, &s->handlectx->mutex);
    pthread_mutex_unlock(&s->handlectx->mutex);
    return ret;
}

void ff_omx_cleanup(OMXCodecContext *s)
{
    int executing;

    pthread_mutex_lock(&s->handlectx->mutex);
    executing = s->state == OMX_StateExecuting;
    s->handlectx->terminating = 1;
    pthread_mutex_unlock(&s->handlectx->mutex);

    if (executing) {
        OMX_SendCommand(s->handle, OMX_CommandStateSet, OMX_StateIdle, NULL);
        ff_omx_wait_for_state(s, OMX_StateIdle);
        OMX_SendCommand(s->handle, OMX_CommandStateSet, OMX_StateLoaded, NULL);

        pthread_mutex_lock(&s->handlectx->mutex);
        while (s->state != OMX_StateLoaded) {
            OMX_BUFFERHEADERTYPE *buffer;

            if (!s->num_free_in_buffers && !s->num_done_out_buffers)
                pthread_cond_wait(&s->handlectx->cond, &s->handlectx->mutex);

            while (buffer = ff_omx_get_buffer_locked(s, &s->num_free_in_buffers, s->free_in_buffers, 0)) {
                pthread_mutex_unlock(&s->handlectx->mutex);
                OMX_FreeBuffer(s->handle, s->in_port, buffer);
                pthread_mutex_lock(&s->handlectx->mutex);
            }
            while (buffer = ff_omx_get_buffer_locked(s, &s->num_done_out_buffers, s->done_out_buffers, 0)) {
                pthread_mutex_unlock(&s->handlectx->mutex);
                OMX_FreeBuffer(s->handle, s->out_port, buffer);
                pthread_mutex_unlock(&s->handlectx->mutex);
            }

            if (s->error == OMX_ErrorNone)
                break;
        }
        pthread_mutex_unlock(&s->handlectx->mutex);
    }

    pthread_mutex_lock(&s->handlectx->mutex);
    s->handlectx->ctx = NULL;
    pthread_mutex_unlock(&s->handlectx->mutex);

    av_buffer_unref(&s->ref);

    av_freep(&s->in_buffer_headers);
    av_freep(&s->out_buffer_headers);
    av_freep(&s->free_in_buffers);
    av_freep(&s->done_out_buffers);
    av_freep(&s->output_buf);
}

static const struct {
	enum AVPixelFormat pixfmt;
	enum OMX_COLOR_FORMATTYPE omxfmt;
} omx_pix_fmt_map[] = {
	{ AV_PIX_FMT_YUV420P, OMX_COLOR_FormatYUV420Planar },
	{ AV_PIX_FMT_NV12,    OMX_COLOR_FormatYUV420SemiPlanar },
	{ AV_PIX_FMT_NONE,    OMX_COLOR_FormatUnused },
};

enum AVPixelFormat ff_omx_get_pix_fmt(enum OMX_COLOR_FORMATTYPE omxfmt)
{
    unsigned i;
    for (i = 0; omx_pix_fmt_map[i].pixfmt != AV_PIX_FMT_NONE; i++) {
        if (omx_pix_fmt_map[i].omxfmt == omxfmt)
            return omx_pix_fmt_map[i].pixfmt;
    }
    return AV_PIX_FMT_NONE;
}

enum OMX_COLOR_FORMATTYPE ff_omx_get_color_format(enum AVPixelFormat pixfmt)
{
    unsigned i;
    for (i = 0; omx_pix_fmt_map[i].pixfmt != AV_PIX_FMT_NONE; i++) {
        if (omx_pix_fmt_map[i].pixfmt == pixfmt)
            return omx_pix_fmt_map[i].omxfmt;
    }
    return OMX_COLOR_FormatUnused;
}
