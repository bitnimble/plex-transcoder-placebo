/*
 * Text subtitle rendering filter, based on vf_subtitles
 *
 * Copyright (c) 2016 Plex, Inc.
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

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <ass/ass.h>
#include "libavcodec/avcodec.h"
#include "libavfilter/avfilter.h"
#include "libavutil/avstring.h"
#include "libavutil/opt.h"
#include "libavformat/avformat.h"
#include "vf_inlineass.h"
#include "drawutils.h"
#include "formats.h"
#include "internal.h"
#include "video.h"
#include "libavutil/frame.h"

typedef struct {
    const AVClass *class;
    ASS_Library *library;
    ASS_Renderer *renderer;
    ASS_Track *track;

    char *font_path;
    char *font_name;
    char *overrides;
    int apply_overrides;
    char *fonts_dir;
    char *fc_file;
    double font_scale;
    double font_size;
    double outline;
    double shadow;
    double marginH, marginV;
    int atsc_cc;
    char *language;

    FFDrawContext draw;

    int got_header;

    int mangle_state;
    float vs_rgb2yuv[3][4];
    float vs2rgb[3][4];

    AVCodecContext *atsc_dec;
} AssContext;

#define OFFSET(x) offsetof(AssContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

#define ASS_TIME_BASE av_make_q(1, 1000)

/* libass supports a log level ranging from 0 to 7 */
static const int ass_libavfilter_log_level_map[] = {
    AV_LOG_QUIET,               /* 0 */
    AV_LOG_PANIC,               /* 1 */
    AV_LOG_FATAL,               /* 2 */
    AV_LOG_ERROR,               /* 3 */
    AV_LOG_WARNING,             /* 4 */
    AV_LOG_INFO,                /* 5 */
    AV_LOG_VERBOSE,             /* 6 */
    AV_LOG_DEBUG,               /* 7 */
};

static void ass_log(int ass_level, const char *fmt, va_list args, void *ctx)
{
    int level = ass_libavfilter_log_level_map[ass_level];

    av_vlog(ctx, level, fmt, args);
    av_log(ctx, level, "\n");
}

static av_cold int init(AVFilterContext *ctx)
{
    AssContext *ass = ctx->priv;

    ass->library = ass_library_init();

    if (!ass->library) {
        av_log(ctx, AV_LOG_ERROR, "ass_library_init() failed!\n");
        return AVERROR(EINVAL);
    }

    ass_set_message_cb(ass->library, ass_log, ctx);
    ass_set_extract_fonts(ass->library, 1);

    if (ass->fonts_dir)
        ass_set_fonts_dir(ass->library, ass->fonts_dir);

    ass->renderer = ass_renderer_init(ass->library);
    if (!ass->renderer) {
        av_log(ctx, AV_LOG_ERROR, "ass_renderer_init() failed!\n");
        return AVERROR(EINVAL);
    }

    ass_set_font_scale(ass->renderer, ass->font_scale);

    ass->track = ass_new_track(ass->library);
    if (!ass->track) {
        av_log(ctx, AV_LOG_ERROR, "ass_new_track() failed!\n");
        return AVERROR(EINVAL);
    }

    if (ass->language)
        ass->track->Language = strdup(ass->language);

    if (ass->atsc_cc) {
        AVDictionary *options = NULL;
        int ret;
        AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_EIA_608);
        if (!codec) {
            av_log(ctx, AV_LOG_ERROR, "failed to find EIA-608 decoder\n");
            return AVERROR_DECODER_NOT_FOUND;
        }

        if (!(ass->atsc_dec = avcodec_alloc_context3(codec))) {
            av_log(ctx, AV_LOG_ERROR, "failed to allocate EIA-608 decoder\n");
            return AVERROR(ENOMEM);
        }

        av_dict_set(&options, "sub_text_format", "ass", 0);
        av_dict_set(&options, "real_time", "1", 0);

        if ((ret = avcodec_open2(ass->atsc_dec, codec, &options)) < 0) {
            av_log(ctx, AV_LOG_ERROR, "failed to open EIA-608 decoder: %i\n", ret);
            return ret;
        }

        av_dict_free(&options);

        avfilter_inlineass_set_fonts(ctx);
    }

    ass->mangle_state = 0;

    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    AssContext *ass = ctx->priv;

    if (ass->track)
        ass_free_track(ass->track);
    if (ass->renderer)
        ass_renderer_done(ass->renderer);
    if (ass->library)
        ass_library_done(ass->library);
}

static int query_formats(AVFilterContext *ctx)
{
    return ff_set_common_formats(ctx, ff_draw_supported_pixel_formats(0));
}

static int config_input(AVFilterLink *link)
{
    AssContext *context = link->dst->priv;
    ff_draw_init(&context->draw, link->format, 0);

    ass_set_frame_size(context->renderer, link->w, link->h);

    ass_set_pixel_aspect(context->renderer, av_q2d(link->sample_aspect_ratio));

    if (context->atsc_dec)
        context->atsc_dec->pkt_timebase = link->time_base;

    return 0;
}

// Bunch of nonsense from MPlayer
#define COL_Y 0
#define COL_U 1
#define COL_V 2
#define COL_C 3
enum mp_csp {
    MP_CSP_AUTO,
    MP_CSP_BT_601,
    MP_CSP_BT_709,
    MP_CSP_SMPTE_240M,
    MP_CSP_RGB,
    MP_CSP_XYZ,
    MP_CSP_YCGCO,
    MP_CSP_COUNT
};

// Any enum mp_csp value is a valid index (except MP_CSP_COUNT)
extern const char *const mp_csp_names[MP_CSP_COUNT];

enum mp_csp_levels {
    MP_CSP_LEVELS_AUTO,
    MP_CSP_LEVELS_TV,
    MP_CSP_LEVELS_PC,
    MP_CSP_LEVELS_COUNT,
};
// initializer for struct mp_csp_details that contains reasonable defaults
#define MP_CSP_DETAILS_DEFAULTS {MP_CSP_BT_601, MP_CSP_LEVELS_TV, MP_CSP_LEVELS_PC}

struct mp_csp_details {
    enum mp_csp format;
    enum mp_csp_levels levels_in;      // encoded video
    enum mp_csp_levels levels_out;     // output device
};

struct mp_csp_params {
    struct mp_csp_details colorspace;
    float brightness;
    float contrast;
    float hue;
    float saturation;
    float rgamma;
    float ggamma;
    float bgamma;
    // texture_bits/input_bits is for rescaling fixed point input to range [0,1]
    int texture_bits;
    int input_bits;
    // for scaling integer input and output (if 0, assume range [0,1])
    int int_bits_in;
    int int_bits_out;
};

#define MP_CSP_PARAMS_DEFAULTS {                                \
    .colorspace = MP_CSP_DETAILS_DEFAULTS,                      \
    .brightness = 0, .contrast = 1, .hue = 0, .saturation = 1,  \
    .rgamma = 1, .ggamma = 1, .bgamma = 1,                      \
    .texture_bits = 8, .input_bits = 8}

/* Fill in the Y, U, V vectors of a yuv2rgb conversion matrix
 * based on the given luma weights of the R, G and B components (lr, lg, lb).
 * lr+lg+lb is assumed to equal 1.
 * This function is meant for colorspaces satisfying the following
 * conditions (which are true for common YUV colorspaces):
 * - The mapping from input [Y, U, V] to output [R, G, B] is linear.
 * - Y is the vector [1, 1, 1].  (meaning input Y component maps to 1R+1G+1B)
 * - U maps to a value with zero R and positive B ([0, x, y], y > 0;
 *   i.e. blue and green only).
 * - V maps to a value with zero B and positive R ([x, y, 0], x > 0;
 *   i.e. red and green only).
 * - U and V are orthogonal to the luma vector [lr, lg, lb].
 * - The magnitudes of the vectors U and V are the minimal ones for which
 *   the image of the set Y=[0...1],U=[-0.5...0.5],V=[-0.5...0.5] under the
 *   conversion function will cover the set R=[0...1],G=[0...1],B=[0...1]
 *   (the resulting matrix can be converted for other input/output ranges
 *   outside this function).
 * Under these conditions the given parameters lr, lg, lb uniquely
 * determine the mapping of Y, U, V to R, G, B.
 */
static void luma_coeffs(float m[3][4], float lr, float lg, float lb)
{
    assert(fabs(lr+lg+lb - 1) < 1e-6);
    m[0][0] = m[1][0] = m[2][0] = 1;
    m[0][1] = 0;
    m[1][1] = -2 * (1-lb) * lb/lg;
    m[2][1] = 2 * (1-lb);
    m[0][2] = 2 * (1-lr);
    m[1][2] = -2 * (1-lr) * lr/lg;
    m[2][2] = 0;
    // Constant coefficients (m[x][3]) not set here
}

/**
 * \brief get the coefficients of the yuv -> rgb conversion matrix
 * \param params struct specifying the properties of the conversion like
 *  brightness, ...
 * \param m array to store coefficients into
 */
static void mp_get_yuv2rgb_coeffs(struct mp_csp_params *params, float m[3][4])
{
    int format = params->colorspace.format;
    if (format <= MP_CSP_AUTO || format >= MP_CSP_COUNT)
        format = MP_CSP_BT_601;
    int levels_in = params->colorspace.levels_in;
    if (levels_in <= MP_CSP_LEVELS_AUTO || levels_in >= MP_CSP_LEVELS_COUNT)
        levels_in = MP_CSP_LEVELS_TV;

    switch (format) {
    case MP_CSP_BT_601:     luma_coeffs(m, 0.299,  0.587,  0.114 ); break;
    case MP_CSP_BT_709:     luma_coeffs(m, 0.2126, 0.7152, 0.0722); break;
    case MP_CSP_SMPTE_240M: luma_coeffs(m, 0.2122, 0.7013, 0.0865); break;
    case MP_CSP_RGB: {
        static const float ident[3][4] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        memcpy(m, ident, sizeof(ident));
        levels_in = -1;
        break;
    }
    case MP_CSP_XYZ: {
        static const float xyz_to_rgb[3][4] = {
            {3.2404542,  -1.5371385, -0.4985314},
            {-0.9692660,  1.8760108,  0.0415560},
            {0.0556434,  -0.2040259,  1.0572252},
        };
        memcpy(m, xyz_to_rgb, sizeof(xyz_to_rgb));
        levels_in = -1;
        break;
    }
    case MP_CSP_YCGCO: {
        static const float ycgco_to_rgb[3][4] = {
            {1,  -1,  1},
            {1,   1,  0},
            {1,  -1, -1},
        };
        memcpy(m, ycgco_to_rgb, sizeof(ycgco_to_rgb));
        break;
    }
    default:
        abort();
    };

    // Hue is equivalent to rotating input [U, V] subvector around the origin.
    // Saturation scales [U, V].
    float huecos = params->saturation * cos(params->hue);
    float huesin = params->saturation * sin(params->hue);
    for (int i = 0; i < 3; i++) {
        float u = m[i][COL_U];
        m[i][COL_U] = huecos * u - huesin * m[i][COL_V];
        m[i][COL_V] = huesin * u + huecos * m[i][COL_V];
    }

    assert(params->input_bits >= 8);
    assert(params->texture_bits >= params->input_bits);
    double s = (1 << (params->input_bits-8)) / ((1<<params->texture_bits)-1.);
    // The values below are written in 0-255 scale
    struct yuvlevels { double ymin, ymax, cmin, cmid; }
        yuvlim =  { 16*s, 235*s, 16*s, 128*s },
        yuvfull = {  0*s, 255*s,  1*s, 128*s },  // '1' for symmetry around 128
        anyfull = {  0*s, 255*s, -255*s/2, 0 },
        yuvlev;
    switch (levels_in) {
    case MP_CSP_LEVELS_TV: yuvlev = yuvlim; break;
    case MP_CSP_LEVELS_PC: yuvlev = yuvfull; break;
    case -1: yuvlev = anyfull; break;
    default:
        abort();
    }

    int levels_out = params->colorspace.levels_out;
    if (levels_out <= MP_CSP_LEVELS_AUTO || levels_out >= MP_CSP_LEVELS_COUNT)
        levels_out = MP_CSP_LEVELS_PC;
    struct rgblevels { double min, max; }
        rgblim =  { 16/255., 235/255. },
        rgbfull = {      0,        1  },
        rgblev;
    switch (levels_out) {
    case MP_CSP_LEVELS_TV: rgblev = rgblim; break;
    case MP_CSP_LEVELS_PC: rgblev = rgbfull; break;
    default:
        abort();
    }

    double ymul = (rgblev.max - rgblev.min) / (yuvlev.ymax - yuvlev.ymin);
    double cmul = (rgblev.max - rgblev.min) / (yuvlev.cmid - yuvlev.cmin) / 2;
    for (int i = 0; i < 3; i++) {
        m[i][COL_Y] *= ymul;
        m[i][COL_U] *= cmul;
        m[i][COL_V] *= cmul;
        // Set COL_C so that Y=umin,UV=cmid maps to RGB=min (black to black)
        m[i][COL_C] = rgblev.min - m[i][COL_Y] * yuvlev.ymin
                      -(m[i][COL_U] + m[i][COL_V]) * yuvlev.cmid;
    }

    // Brightness adds a constant to output R,G,B.
    // Contrast scales Y around 1/2 (not 0 in this implementation).
    for (int i = 0; i < 3; i++) {
        m[i][COL_C] += params->brightness;
        m[i][COL_Y] *= params->contrast;
        m[i][COL_C] += (rgblev.max-rgblev.min) * (1 - params->contrast)/2;
    }

    int in_bits = FFMAX(params->int_bits_in, 1);
    int out_bits = FFMAX(params->int_bits_out, 1);
    double in_scale = (1 << in_bits) - 1.0;
    double out_scale = (1 << out_bits) - 1.0;
    for (int i = 0; i < 3; i++) {
        m[i][COL_C] *= out_scale; // constant is 1.0
        for (int x = 0; x < 3; x++)
            m[i][x] *= out_scale / in_scale;
    }
}

static void mp_invert_yuv2rgb(float out[3][4], float in[3][4])
{
    float m00 = in[0][0], m01 = in[0][1], m02 = in[0][2], m03 = in[0][3],
          m10 = in[1][0], m11 = in[1][1], m12 = in[1][2], m13 = in[1][3],
          m20 = in[2][0], m21 = in[2][1], m22 = in[2][2], m23 = in[2][3];

    // calculate the adjoint
    out[0][0] =  (m11 * m22 - m21 * m12);
    out[0][1] = -(m01 * m22 - m21 * m02);
    out[0][2] =  (m01 * m12 - m11 * m02);
    out[1][0] = -(m10 * m22 - m20 * m12);
    out[1][1] =  (m00 * m22 - m20 * m02);
    out[1][2] = -(m00 * m12 - m10 * m02);
    out[2][0] =  (m10 * m21 - m20 * m11);
    out[2][1] = -(m00 * m21 - m20 * m01);
    out[2][2] =  (m00 * m11 - m10 * m01);

    // calculate the determinant (as inverse == 1/det * adjoint,
    // adjoint * m == identity * det, so this calculates the det)
    float det = m00 * out[0][0] + m10 * out[0][1] + m20 * out[0][2];
    det = 1.0f / det;

    out[0][0] *= det;
    out[0][1] *= det;
    out[0][2] *= det;
    out[1][0] *= det;
    out[1][1] *= det;
    out[1][2] *= det;
    out[2][0] *= det;
    out[2][1] *= det;
    out[2][2] *= det;

    // fix the constant coefficient
    // rgb = M * yuv + C
    // M^-1 * rgb = yuv + M^-1 * C
    // yuv = M^-1 * rgb - M^-1 * C
    //                  ^^^^^^^^^^
    out[0][3] = -(out[0][0] * m03 + out[0][1] * m13 + out[0][2] * m23);
    out[1][3] = -(out[1][0] * m03 + out[1][1] * m13 + out[1][2] * m23);
    out[2][3] = -(out[2][0] * m03 + out[2][1] * m13 + out[2][2] * m23);
}

// Multiply the color in c with the given matrix.
// c is {R, G, B} or {Y, U, V} (depending on input/output and matrix).
// Output is clipped to the given number of bits.
static void mp_map_int_color(float matrix[3][4], int clip_bits, int c[3])
{
    int in[3] = {c[0], c[1], c[2]};
    for (int i = 0; i < 3; i++) {
        double val = matrix[i][3];
        for (int x = 0; x < 3; x++)
            val += matrix[i][x] * in[x];
        int ival = lrint(val);
        c[i] = av_clip(ival, 0, (1 << clip_bits) - 1);
    }
}

// Disgusting hack for (xy-)vsfilter color compatibility.
// From mpv
static void calculate_mangle_table(AssContext *ass, AVFrame *frame)
{
    int out_csp = frame->colorspace;
    int out_levels = frame->color_range;
    int csp = 0;
    int levels = 0;
    const ASS_Track *track = ass->track;
    static const int ass_csp[] = {
        [YCBCR_BT601_TV]        = MP_CSP_BT_601,
        [YCBCR_BT601_PC]        = MP_CSP_BT_601,
        [YCBCR_BT709_TV]        = MP_CSP_BT_709,
        [YCBCR_BT709_PC]        = MP_CSP_BT_709,
        [YCBCR_SMPTE240M_TV]    = MP_CSP_SMPTE_240M,
        [YCBCR_SMPTE240M_PC]    = MP_CSP_SMPTE_240M,
    };
    static const int av_csp[] = {
        [AVCOL_SPC_RGB]          = MP_CSP_RGB,
        [AVCOL_SPC_BT709]        = MP_CSP_BT_601, // We do something dumb here for FFDraw compatibility
        [AVCOL_SPC_UNSPECIFIED]  = MP_CSP_AUTO,
        [AVCOL_SPC_FCC]          = MP_CSP_AUTO,
        [AVCOL_SPC_BT470BG]      = MP_CSP_BT_601,
        [AVCOL_SPC_SMPTE170M]    = MP_CSP_BT_601,
        [AVCOL_SPC_SMPTE240M]    = MP_CSP_SMPTE_240M,
    };
    static const int ass_levels[] = {
        [YCBCR_BT601_TV]        = MP_CSP_LEVELS_TV,
        [YCBCR_BT601_PC]        = MP_CSP_LEVELS_PC,
        [YCBCR_BT709_TV]        = MP_CSP_LEVELS_TV,
        [YCBCR_BT709_PC]        = MP_CSP_LEVELS_PC,
        [YCBCR_SMPTE240M_TV]    = MP_CSP_LEVELS_TV,
        [YCBCR_SMPTE240M_PC]    = MP_CSP_LEVELS_PC,
    };
    static const int av_levels[] = {
        [AVCOL_RANGE_UNSPECIFIED] = MP_CSP_LEVELS_AUTO,
        [AVCOL_RANGE_MPEG]        = MP_CSP_LEVELS_TV,
        [AVCOL_RANGE_JPEG]        = MP_CSP_LEVELS_PC,
    };
    int trackcsp = track->YCbCrMatrix;
    // NONE is a bit random, but the intention is: don't modify colors.
    if (trackcsp == YCBCR_NONE) {
        ass->mangle_state = 2;
        return;
    }
    if (trackcsp < sizeof(ass_csp) / sizeof(ass_csp[0]))
        csp = ass_csp[trackcsp];
    if (out_csp < sizeof(av_csp) / sizeof(av_csp[0]))
        out_csp = av_csp[out_csp];
    if (trackcsp < sizeof(ass_levels) / sizeof(ass_levels[0]))
        levels = ass_levels[trackcsp];
    if (out_levels < sizeof(av_levels) / sizeof(av_levels[0]))
        out_levels = av_levels[out_levels];
    if (trackcsp == YCBCR_DEFAULT) {
        csp = MP_CSP_BT_601;
        levels = MP_CSP_LEVELS_TV;
    }
    // Unknown colorspace (either YCBCR_UNKNOWN, or a valid value unknown to us)
    if (!csp || !levels || !out_csp || !out_levels || out_csp == MP_CSP_AUTO ||
        (csp == out_csp && levels == out_levels)) {
        ass->mangle_state = 2;
        return;
    }

    // Conversion that VSFilter would use
    struct mp_csp_params vs_params = MP_CSP_PARAMS_DEFAULTS;
    vs_params.colorspace.format = csp;
    vs_params.colorspace.levels_in = levels;
    vs_params.int_bits_in = 8;
    vs_params.int_bits_out = 8;
    float vs_yuv2rgb[3][4];
    mp_get_yuv2rgb_coeffs(&vs_params, vs_yuv2rgb);
    mp_invert_yuv2rgb(ass->vs_rgb2yuv, vs_yuv2rgb);

    // Proper conversion to RGB
    struct mp_csp_params rgb_params = MP_CSP_PARAMS_DEFAULTS;
    rgb_params.colorspace.format = out_csp;
    rgb_params.colorspace.levels_in = out_levels;
    rgb_params.int_bits_in = 8;
    rgb_params.int_bits_out = 8;
    mp_get_yuv2rgb_coeffs(&rgb_params, ass->vs2rgb);

    ass->mangle_state = 1;
}

/* libass stores an RGBA color in the format RRGGBBTT, where TT is the transparency level */
#define AR(c)  ( (c)>>24)
#define AG(c)  (((c)>>16)&0xFF)
#define AB(c)  (((c)>>8) &0xFF)
#define AA(c)  ((0xFF-c) &0xFF)

static void overlay_ass_image(AssContext *ass, AVFrame *picref,
                              const ASS_Image *image)
{
    uint8_t rgba_color[4];
    for (; image; image = image->next) {
        if (ass->mangle_state == 1) {
            int c[3] = {AR(image->color), AG(image->color), AB(image->color)};
            mp_map_int_color(ass->vs_rgb2yuv, 8, c);
            mp_map_int_color(ass->vs2rgb, 8, c);

            rgba_color[0] = c[0];
            rgba_color[1] = c[1];
            rgba_color[2] = c[2];
            rgba_color[3] = AA(image->color);
        } else {
            rgba_color[0] = AR(image->color);
            rgba_color[1] = AG(image->color);
            rgba_color[2] = AB(image->color);
            rgba_color[3] = AA(image->color);
        }
        FFDrawColor color;
        ff_draw_color(&ass->draw, &color, rgba_color);
        ff_blend_mask(&ass->draw, &color,
                      picref->data, picref->linesize,
                      picref->width, picref->height,
                      image->bitmap, image->stride, image->w, image->h,
                      3, 0, image->dst_x, image->dst_y);
    }
}

static int handle_atsc_cc(AVFilterLink *inlink, AVFrame *picref)
{
    AVFrameSideData *sd = av_frame_get_side_data(picref, AV_FRAME_DATA_A53_CC);
    AVFilterContext *ctx = inlink->dst;
    AssContext *ass = ctx->priv;

    if (!sd)
        return 0;

    int ret = 0;
    AVPacket pkt = {0};
    AVSubtitle sub = {0};
    int got_output = 0;

    av_init_packet(&pkt);
    if (!(pkt.buf = av_buffer_ref(sd->buf)))
        return AVERROR(ENOMEM);
    pkt.data = sd->data;
    pkt.size = sd->size;
    pkt.pts  = picref->pts;

    if (avcodec_decode_subtitle2(ass->atsc_dec, &sub, &got_output, &pkt) < 0)
        goto fail;

    if (!got_output)
        goto fail;

    ass_flush_events(ass->track);

    avfilter_inlineass_append_data(ctx, ass->atsc_dec, &sub);

fail:
    av_frame_remove_side_data(picref, AV_FRAME_DATA_A53_CC);
    av_packet_unref(&pkt);
    avsubtitle_free(&sub);
    return ret;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *picref)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    AssContext *ass = ctx->priv;
    ASS_Image *image = NULL;
    long long time_ms = av_rescale_q(picref->pts, inlink->time_base, ASS_TIME_BASE);

    if (!ass->mangle_state) {
        calculate_mangle_table(ass, picref);
    }

    if (ass->atsc_dec)
        handle_atsc_cc(inlink, picref);

    image = ass_render_frame(ass->renderer, ass->track, time_ms, NULL);

    overlay_ass_image(ass, picref, image);

    return ff_filter_frame(outlink, picref);
}

static const AVFilterPad inlineass_inputs[] = {
    {
        .name             = "default",
        .type             = AVMEDIA_TYPE_VIDEO,
        .filter_frame     = filter_frame,
        .config_props     = config_input,
        .needs_writable   = 1,
    },
    { NULL }
};

static const AVFilterPad inlineass_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

void avfilter_inlineass_set_storage_size(AVFilterContext *context, int w, int h)
{
    AssContext *ass = (AssContext *)context->priv;
    ass_set_storage_size(ass->renderer, w, h);
}

void avfilter_inlineass_add_attachment(AVFilterContext *context, AVStream *st)
{
    AVDictionaryEntry *e = NULL;
    char *filename = NULL, *ext = NULL;
    AssContext *assContext = (AssContext*)context->priv;
    if (!st->codecpar->extradata_size)
        return;
    e = av_dict_get(st->metadata, "filename", NULL, 0);
    if (!e)
        return; // Nobody adds fonts without filenames anyway
    filename = e->value;
    ext = filename + strlen(filename) - 4;
    if (st->codecpar->codec_id == AV_CODEC_ID_TTF ||
        st->codecpar->codec_id == AV_CODEC_ID_OTF ||
        !av_strcasecmp( ext, ".ttf" ) ||
        !av_strcasecmp( ext, ".otf" ) ||
        !av_strcasecmp( ext, ".ttc" )
    ) {
        ass_add_font(assContext->library, filename,
                     st->codecpar->extradata, st->codecpar->extradata_size);
    }
}

void avfilter_inlineass_set_fonts(AVFilterContext *context)
{
    AssContext* ass = context->priv;
    ass_set_fonts(ass->renderer, ass->font_path, "DejaVu Sans", 1, ass->fc_file, 1);
}

static int process_header(AVFilterContext *link, AVCodecContext *dec_ctx)
{
    int ret = 0;
    AssContext *ass = link->priv;
    ASS_Track *track = ass->track;
    enum AVCodecID codecID = dec_ctx->codec_id;

    if (!track)
        return AVERROR(EINVAL);

    if (codecID == AV_CODEC_ID_ASS) {
        ass_process_codec_private(track, dec_ctx->extradata,
                                  dec_ctx->extradata_size);
    } else {
        /* Decode subtitles and push them into the renderer (libass) */
        if (dec_ctx->subtitle_header)
            ass_process_codec_private(track,
                                      dec_ctx->subtitle_header,
                                      dec_ctx->subtitle_header_size);

        if (!ass->track->n_styles) {
            int sid = ass_alloc_style(track);
            if (sid < 0)
                return AVERROR(ENOMEM);
            ASS_Style *style = &ass->track->styles[sid];
            style = &ass->track->styles[sid];
            style->Name             = strdup("Default");
            style->FontName         = strdup(ass->font_name);
            style->FontSize         = ass->font_size * track->PlayResY / 720.;
            style->PrimaryColour    = 0xffffff00U;
            style->SecondaryColour  = 0x00ffff00U;
            style->OutlineColour    = 0x13070200U;
            style->BackColour       = 0x00000033U;
            style->Bold             = 500;
            style->ScaleX           = 1.0;
            style->ScaleY           = 1.0;
            style->Spacing          = 0;
            style->BorderStyle      = 1;
            style->Outline          = ass->outline * track->PlayResY / 720.;
            style->Shadow           = ass->shadow * track->PlayResY / 720.;
            style->Alignment        = 2;
            style->MarginL = style->MarginR = ass->marginH * track->PlayResY / 720.;
            style->MarginV = ass->marginV * track->PlayResY / 720.;

            track->default_style = sid;
        }
    }

    if (ass->apply_overrides && (codecID != AV_CODEC_ID_ASS || ass->apply_overrides > 0)) {
        const char *inp = ass->overrides;
        char** list = NULL;
        int list_nb = 0;

        while (inp && inp[0]) {
            char* tok = av_get_token(&inp, ",");
            if (!tok)
                goto fail;

            if ((ret = av_dynarray_add_nofree(&list, &list_nb, tok)) < 0) {
                av_free(tok);
                goto fail;
            }

            if (*inp)
                inp++;
        }

#define ADD(a, b, c) do { \
    char* tok = av_asprintf(#a "=" b, (c)); \
    if (!tok) \
        goto fail; \
    ret = av_dynarray_add_nofree(&list, &list_nb, tok); \
    if (ret < 0) { \
        av_free(tok); \
        goto fail; \
    } \
} while (0)

#define ADD2(a, b) ADD(a, "%f", ((ass->b) * (double)track->PlayResY / 720.))

        ADD2(Outline, outline);
        ADD2(Shadow, shadow);
        ADD2(FontSize, font_size);
        ADD2(MarginL, marginH);
        ADD2(MarginR, marginH);
        ADD2(MarginV, marginV);

        if ((ret = av_dynarray_add_nofree(&list, &list_nb, NULL)) < 0)
            goto fail;

        ass_set_style_overrides(ass->library, list);

fail:
        for (int i = 0; i < list_nb; i++)
            av_free(list[i]);
        av_free(list);

        ass_process_force_style(track);
    }

    ass->got_header = 1;

    return ret;
}

void avfilter_inlineass_append_data(AVFilterContext *link, AVCodecContext *dec_ctx,
                                    AVSubtitle *sub)
{
    AssContext *ass = link->priv;
    int i;

    if (!ass->got_header)
        process_header(link, dec_ctx);

    for (i = 0; i < sub->num_rects; i++) {
        int64_t duration = sub->end_display_time - sub->start_display_time;
        int64_t start = av_rescale_q(sub->pts, AV_TIME_BASE_Q, ASS_TIME_BASE);
        start        += sub->start_display_time;
        char *ass_line = sub->rects[i]->ass;
        if (!ass_line)
            break;
        ass_process_chunk(ass->track, ass_line, strlen(ass_line), start, duration);
    }
}

#define DEFAULT_OVERRIDES \
    "ScaledBorderAndShadow=yes," \
    "FontName=Noto Sans Medium," \
    "Bold=500," \
    "PrimaryColour=&H00FFFFFF," \
    "OutlineColour=&H00020713," \
    "BackColour=&HCC000000" \

static const AVOption inlineass_options[] = {
    {"font_scale",     "font scale factor",                OFFSET(font_scale), AV_OPT_TYPE_DOUBLE, {.dbl = 1.0 }, 0.0f,      100.0f,   FLAGS},
    {"font_path",      "path to default font",             OFFSET(font_path),  AV_OPT_TYPE_STRING, {.str = NULL}, CHAR_MIN,  CHAR_MAX, FLAGS},
    {"font_name",      "name of default font",             OFFSET(font_name),  AV_OPT_TYPE_STRING, {.str = "Noto Sans Medium"}, CHAR_MIN,  CHAR_MAX, FLAGS},
    {"overrides",      "style overrides",                  OFFSET(overrides),  AV_OPT_TYPE_STRING, {.str = DEFAULT_OVERRIDES}, CHAR_MIN,  CHAR_MAX, FLAGS},
    {"apply_overrides","whether to apply style overrides", OFFSET(apply_overrides),AV_OPT_TYPE_BOOL,{.i64 = -1 }, -1,        1,        FLAGS},
    {"font_size",      "default font size",                OFFSET(font_size),  AV_OPT_TYPE_DOUBLE, {.dbl = 54},   0.0f,      100.0f,   FLAGS},
    {"outline",        "default outline thickness",        OFFSET(outline),    AV_OPT_TYPE_DOUBLE, {.dbl = 2.6},  0.0f,      100.0f,   FLAGS},
    {"shadow",         "default shadow offset",            OFFSET(shadow),     AV_OPT_TYPE_DOUBLE, {.dbl = 1.7},  0.0f,      100.0f,   FLAGS},
    {"marginH",        "default horizontal margin",        OFFSET(marginH),    AV_OPT_TYPE_DOUBLE, {.dbl = 100},  0.0f,      6400.0f,  FLAGS},
    {"marginV",        "default vertical margin",          OFFSET(marginV),    AV_OPT_TYPE_DOUBLE, {.dbl = 42 },  0.0f,      320.0f,   FLAGS},
    {"fonts_dir",      "directory to scan for fonts",      OFFSET(fonts_dir),  AV_OPT_TYPE_STRING, {.str = NULL}, CHAR_MIN,  CHAR_MAX, FLAGS},
    {"fontconfig_file","fontconfig file to load",          OFFSET(fc_file),    AV_OPT_TYPE_STRING, {.str = NULL}, CHAR_MIN,  CHAR_MAX, FLAGS},
    {"atsc_cc",        "burn ATSC closed captions",        OFFSET(atsc_cc),    AV_OPT_TYPE_BOOL,   {.i64 = 0   }, 0,         1,        FLAGS},
    {"language",       "default language",                 OFFSET(language),   AV_OPT_TYPE_STRING, {.str = NULL}, CHAR_MIN,  CHAR_MAX, FLAGS},
    {NULL},
};

AVFILTER_DEFINE_CLASS(inlineass);

AVFilter ff_vf_inlineass ={
    .name          = "inlineass",
    .description   = NULL_IF_CONFIG_SMALL("Render subtitles onto input video using the libass library."),
    .priv_size     = sizeof(AssContext),
    .priv_class    = &inlineass_class,
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .inputs        = inlineass_inputs,
    .outputs       = inlineass_outputs,
  };
