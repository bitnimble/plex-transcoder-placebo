/*
 * Copyright (c) 2017 Vittorio Giovara <vittorio.giovara@gmail.com>
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

/**
 * @file
 * tonemap algorithms
 */

#include <float.h>
#include <stdio.h>
#include <string.h>

#include "libavutil/imgutils.h"
#include "libavutil/internal.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "colorspace.h"
#include "formats.h"
#include "internal.h"
#include "video.h"

#include "tonemap.h"

static const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_GBRPF32,
    AV_PIX_FMT_GBRAPF32,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_NONE,
};

typedef struct ThreadData {
    AVFrame *in, *out;
    const AVPixFmtDescriptor *desc, *odesc;
    double peak;
} ThreadData;

static int query_formats(AVFilterContext *ctx)
{
    AVFilterFormats *formats = ff_make_format_list(pix_fmts);
    int res = ff_formats_ref(formats, &ctx->inputs[0]->outcfg.formats);
    if (res < 0)
        return res;
    formats = NULL;
    res = ff_add_format(&formats, AV_PIX_FMT_NV12);
    if (res < 0)
        return res;
    return ff_formats_ref(formats, &ctx->outputs[0]->incfg.formats);
}

static av_cold int init(AVFilterContext *ctx)
{
    TonemapContext *s = ctx->priv;

    switch(s->tonemap) {
    case TONEMAP_GAMMA:
        if (isnan(s->param))
            s->param = 1.8f;
        break;
    case TONEMAP_REINHARD:
        if (!isnan(s->param))
            s->param = (1.0f - s->param) / s->param;
        break;
    case TONEMAP_MOBIUS:
        if (isnan(s->param))
            s->param = 0.3f;
        break;
    }

    if (isnan(s->param))
        s->param = 1.0f;

    s->tonemap_frame_p010_nv12 = ff_tonemap_frame_p010_nv12_c;

    return 0;
}

static float hable(float in)
{
    float a = 0.15f, b = 0.50f, c = 0.10f, d = 0.20f, e = 0.02f, f = 0.30f;
    return (in * (in * a + b * c) + d * e) / (in * (in * a + b) + d * f) - e / f;
}

static float mobius(float in, float j, double peak)
{
    float a, b;

    if (in <= j)
        return in;

    a = -j * j * (peak - 1.0f) / (j * j - 2.0f * j + peak);
    b = (j * j - 2.0f * j * peak + peak) / FFMAX(peak - 1.0f, 1e-6);

    return (b * b + 2.0f * b * j + j * j) / (b - a) * (in + a) / (in + b);
}

static float eotf_st2084(float x) {
#define ST2084_MAX_LUMINANCE 10000.0f
#define REFERENCE_WHITE 100.0f
#define ST2084_M1 0.1593017578125f
#define ST2084_M2 78.84375f
#define ST2084_C1 0.8359375f
#define ST2084_C2 18.8515625f
#define ST2084_C3 18.6875f

    float p = powf(x, 1.0f / ST2084_M2);
    float a = FFMAX(p -ST2084_C1, 0.0f);
    float b = FFMAX(ST2084_C2 - ST2084_C3 * p, 1e-6f);
    float c  = powf(a / b, 1.0f / ST2084_M1);
    return x > 0.0f ? c * ST2084_MAX_LUMINANCE / REFERENCE_WHITE : 0.0f;
}

static float inverse_eotf_st2084(float x) {
    x *= REFERENCE_WHITE / ST2084_MAX_LUMINANCE;
    x = powf(x, ST2084_M1);
    x = (ST2084_C1 + ST2084_C2 * x) / (1.0f + ST2084_C3 * x);
    return powf(x, ST2084_M2);
}

static float bt2390(float sig_orig, double peak)
{
    float sig_pq = sig_orig / peak;
    float maxLum = 0.751829f / peak; // SDR peak in PQ

    float ks = 1.5f * maxLum - 0.5f;
    float tb = (sig_pq - ks) / (1.0f - ks);
    float tb2 = tb * tb;
    float tb3 = tb2 * tb;
    float pb = (2.0f * tb3 - 3.0f * tb2 + 1.0f) * ks +
               (tb3 - 2.0f * tb2 + tb) * (1.0f - ks) +
               (-2.0f * tb3 + 3.0f * tb2) * maxLum;
    float sig = (sig_pq < ks) ? sig_pq : pb;
    return eotf_st2084(sig * peak);
}

static float mapsig(enum TonemapAlgorithm alg, float sig, double peak, double param)
{
    switch(alg) {
    default:
    case TONEMAP_NONE:
        // do nothing
        break;
    case TONEMAP_LINEAR:
        sig = sig * param / peak;
        break;
    case TONEMAP_GAMMA:
        sig = sig > 0.05f ? pow(sig / peak, 1.0f / param)
                          : sig * pow(0.05f / peak, 1.0f / param) / 0.05f;
        break;
    case TONEMAP_CLIP:
        sig = av_clipf(sig * param, 0, 1.0f);
        break;
    case TONEMAP_HABLE:
        sig = hable(sig) / hable(peak);
        break;
    case TONEMAP_REINHARD:
        sig = sig / (sig + param) * (peak + param) / peak;
        break;
    case TONEMAP_MOBIUS:
        sig = mobius(sig, param, peak);
        break;
    case TONEMAP_BT2390:
        sig = bt2390(sig, peak);
        break;
    }

    return sig;
}

#define MIX(x,y,a) (x) * (1 - (a)) + (y) * (a)
static void tonemap(float r_in, float g_in, float b_in,
                    float *r_out, float *g_out, float *b_out,
                    double param, double desat, double peak,
                    const struct LumaCoefficients *coeffs,
                    enum TonemapAlgorithm alg)
{
    float sig, sig_orig;

    /* load values */
    *r_out = r_in;
    *g_out = g_in;
    *b_out = b_in;

    /* desaturate to prevent unnatural colors */
    if (desat > 0) {
        float luma = coeffs->cr * r_in + coeffs->cg * g_in + coeffs->cb * b_in;
        float overbright = FFMAX(luma - desat, 1e-6) / FFMAX(luma, 1e-6);
        *r_out = MIX(r_in, luma, overbright);
        *g_out = MIX(g_in, luma, overbright);
        *b_out = MIX(b_in, luma, overbright);
    }

    /* pick the brightest component, reducing the value range as necessary
     * to keep the entire signal in range and preventing discoloration due to
     * out-of-bounds clipping */
    sig = FFMAX(FFMAX3(*r_out, *g_out, *b_out), 1e-6);
    sig_orig = sig;

    sig = mapsig(alg, sig, peak, param);

    /* apply the computed scale factor to the color,
     * linearly to prevent discoloration */
    *r_out *= sig / sig_orig;
    *g_out *= sig / sig_orig;
    *b_out *= sig / sig_orig;
}

static void tonemap_int16(int16_t r_in, int16_t g_in, int16_t b_in,
                          int16_t *r_out, int16_t *g_out, int16_t *b_out,
                          float *lin_lut, float *tonemap_lut, uint16_t *delin_lut,
                          const struct LumaCoefficients *coeffs,
                          const struct LumaCoefficients *ocoeffs, double desat,
                          double (*rgb2rgb)[3][3])
{
    int16_t sig;

    /* load values */
    *r_out = r_in;
    *g_out = g_in;
    *b_out = b_in;

    /* pick the brightest component, reducing the value range as necessary
     * to keep the entire signal in range and preventing discoloration due to
     * out-of-bounds clipping */
    sig = FFMAX3(r_in, g_in, b_in);

    float mapval = tonemap_lut[av_clip_uintp2(sig + 2048, 15)];

    float r_lin = lin_lut[av_clip_uintp2(r_in + 2048, 15)];
    float g_lin = lin_lut[av_clip_uintp2(g_in + 2048, 15)];
    float b_lin = lin_lut[av_clip_uintp2(b_in + 2048, 15)];

    r_lin *= mapval;
    g_lin *= mapval;
    b_lin *= mapval;

    /* desaturate to prevent unnatural colors */
    if (desat > 0) {
        float luma = coeffs->cr * r_lin + coeffs->cg * g_lin + coeffs->cb * b_lin;
        float overbright = FFMAX(luma - desat, 1e-6) / FFMAX(luma, 1e-6);
        r_lin = MIX(r_lin, luma, overbright);
        g_lin = MIX(g_lin, luma, overbright);
        b_lin = MIX(b_lin, luma, overbright);
    }

    r_lin = (*rgb2rgb)[0][0] * r_lin + (*rgb2rgb)[0][1] * g_lin + (*rgb2rgb)[0][2] * b_lin;
    g_lin = (*rgb2rgb)[1][0] * r_lin + (*rgb2rgb)[1][1] * g_lin + (*rgb2rgb)[1][2] * b_lin;
    b_lin = (*rgb2rgb)[2][0] * r_lin + (*rgb2rgb)[2][1] * g_lin + (*rgb2rgb)[2][2] * b_lin;

    /*float cmin = FFMIN(FFMIN(r_lin, g_lin), b_lin);
    if (cmin < 0.0) {
        float luma = ocoeffs->cr * r_lin + ocoeffs->cg * g_lin + ocoeffs->cb * b_lin;
        float coeff = cmin / (cmin - luma);
        r_lin = MIX(r_lin, luma, coeff);
        g_lin = MIX(g_lin, luma, coeff);
        b_lin = MIX(b_lin, luma, coeff);
    }
    float cmax = FFMAX(FFMAX(r_lin, g_lin), b_lin);
    if (cmax > 1.0) {
        r_lin /= cmax;
        g_lin /= cmax;
        b_lin /= cmax;
    }*/

    *r_out = delin_lut[av_clip_uintp2(r_lin * 32767 + 0.5, 15)];
    *g_out = delin_lut[av_clip_uintp2(g_lin * 32767 + 0.5, 15)];
    *b_out = delin_lut[av_clip_uintp2(b_lin * 32767 + 0.5, 15)];
}

void ff_tonemap_frame_p010_nv12_c(uint8_t *dsty, uint8_t *dstuv, const uint16_t *srcy, const uint16_t *srcuv, const int *dstlinesize, const int *srclinesize, int width, int height, const struct TonemapIntParams *params)
{
    const int in_depth = 10;
    const int in_uv_offset = 128 << (in_depth - 8);
    const int in_sh = in_depth - 1;
    const int in_rnd = 1 << (in_sh - 1);
    const int in_sh2 = 16 - in_depth;
    const int out_depth = 8;
    const int out_uv_offset = 128 << (out_depth - 8);
    const int out_sh = 29 - out_depth;
    const int out_rnd = 1 << (out_sh - 1);
    int cy  = (*params->yuv2rgb_coeffs)[0][0][0];
    int crv = (*params->yuv2rgb_coeffs)[0][2][0];
    int cgu = (*params->yuv2rgb_coeffs)[1][1][0];
    int cgv = (*params->yuv2rgb_coeffs)[1][2][0];
    int cbu = (*params->yuv2rgb_coeffs)[2][1][0];

    int cry   = (*params->rgb2yuv_coeffs)[0][0][0];
    int cgy   = (*params->rgb2yuv_coeffs)[0][1][0];
    int cby   = (*params->rgb2yuv_coeffs)[0][2][0];
    int cru   = (*params->rgb2yuv_coeffs)[1][0][0];
    int ocgu  = (*params->rgb2yuv_coeffs)[1][1][0];
    int cburv = (*params->rgb2yuv_coeffs)[1][2][0];
    int ocgv  = (*params->rgb2yuv_coeffs)[2][1][0];
    int cbv   = (*params->rgb2yuv_coeffs)[2][2][0];

    int16_t r[4], g[4], b[4];
    for (; height > 1; height -= 2,
                       dsty += dstlinesize[0] * 2, dstuv += dstlinesize[1],
                       srcy += srclinesize[0], srcuv += srclinesize[1] / 2) {
        for (int x = 0; x < width; x += 2) {
            int y00 = (srcy[x]                          >> in_sh2) - params->in_yuv_off;
            int y01 = (srcy[x + 1]                      >> in_sh2) - params->in_yuv_off;
            int y10 = (srcy[srclinesize[0] / 2 + x]     >> in_sh2) - params->in_yuv_off;
            int y11 = (srcy[srclinesize[0] / 2 + x + 1] >> in_sh2) - params->in_yuv_off;
            int u = (srcuv[x]     >> in_sh2) - in_uv_offset,
                v = (srcuv[x + 1] >> in_sh2) - in_uv_offset;

            r[0] = av_clip_int16((y00 * cy + crv * v + in_rnd) >> in_sh);
            r[1] = av_clip_int16((y01 * cy + crv * v + in_rnd) >> in_sh);
            r[2] = av_clip_int16((y10 * cy + crv * v + in_rnd) >> in_sh);
            r[3] = av_clip_int16((y11 * cy + crv * v + in_rnd) >> in_sh);

            g[0] = av_clip_int16((y00 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g[1] = av_clip_int16((y01 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g[2] = av_clip_int16((y10 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);
            g[3] = av_clip_int16((y11 * cy + cgu * u + cgv * v + in_rnd) >> in_sh);

            b[0] = av_clip_int16((y00 * cy + cbu * u + in_rnd) >> in_sh);
            b[1] = av_clip_int16((y01 * cy + cbu * u + in_rnd) >> in_sh);
            b[2] = av_clip_int16((y10 * cy + cbu * u + in_rnd) >> in_sh);
            b[3] = av_clip_int16((y11 * cy + cbu * u + in_rnd) >> in_sh);

            tonemap_int16(r[0], g[0], b[0], &r[0], &g[0], &b[0],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs);
            tonemap_int16(r[1], g[1], b[1], &r[1], &g[1], &b[1],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs);
            tonemap_int16(r[2], g[2], b[2], &r[2], &g[2], &b[2],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs);
            tonemap_int16(r[3], g[3], b[3], &r[3], &g[3], &b[3],
                          params->lin_lut, params->tonemap_lut, params->delin_lut,
                          params->coeffs, params->ocoeffs, params->desat, params->rgb2rgb_coeffs);


            int r00 = r[0], g00 = g[0], b00 = b[0];
            int r01 = r[1], g01 = g[1], b01 = b[1];
            int r10 = r[2], g10 = g[2], b10 = b[2];
            int r11 = r[3], g11 = g[3], b11 = b[3];

            dsty[x]                      = av_clip_uint8(params->out_yuv_off +
                                                 ((r00 * cry + g00 * cgy +
                                                   b00 * cby + out_rnd) >> out_sh));
            dsty[x + 1]                  = av_clip_uint8(params->out_yuv_off +
                                                 ((r01 * cry + g01 * cgy +
                                                   b01 * cby + out_rnd) >> out_sh));
            dsty[x + dstlinesize[0]]     = av_clip_uint8(params->out_yuv_off +
                                                 ((r10 * cry + g10 * cgy +
                                                   b10 * cby + out_rnd) >> out_sh));
            dsty[x + dstlinesize[0] + 1] = av_clip_uint8(params->out_yuv_off +
                                                 ((r11 * cry + g11 * cgy +
                                                   b11 * cby + out_rnd) >> out_sh));


#define avg(a,b,c,d) (((a) + (b) + (c) + (d) + 2) >> 2)
            dstuv[x]     = av_clip_uint8(out_uv_offset +
                                         ((avg(r00, r01, r10, r11) * cru +
                                           avg(g00, g01, g10, g11) * ocgu +
                                           avg(b00, b01, b10, b11) * cburv + out_rnd) >> out_sh));
            dstuv[x + 1] = av_clip_uint8(out_uv_offset +
                                         ((avg(r00, r01, r10, r11) * cburv +
                                           avg(g00, g01, g10, g11) * ocgv +
                                           avg(b00, b01, b10, b11) * cbv + out_rnd) >> out_sh));
#undef avg
        }
    }
}

static float inverse_eotf_bt1886(float c) {
    return c < 0.0f ? 0.0f : powf(c, 1.0f / 2.4f);
}

static int comput_trc_luts(TonemapContext *s, enum AVColorTransferCharacteristic in,
                           enum AVColorTransferCharacteristic out)
{
    int i;

    if (!s->lin_lut && !(s->lin_lut = av_calloc(32768, sizeof(float))))
        return AVERROR(ENOMEM);
    if (!s->delin_lut && !(s->delin_lut = av_calloc(32768, sizeof(uint16_t))))
        return AVERROR(ENOMEM);

    for (i = 0; i < 32768; i++) {
        double v1 = (i - 2048.0) / 28672.0;
        double v2 = i / 32767.0;
        s->lin_lut[i] = FFMAX(eotf_st2084(v1), 0);
        s->delin_lut[i] = av_clip_int16(lrint(inverse_eotf_bt1886(v2) * 28672.0));
    }

    return 0;
}

static int compute_tonemap_lut(TonemapContext *s)
{
    int i;
    double peak = s->lut_peak;

    if (!s->tonemap_lut && !(s->tonemap_lut = av_calloc(32768, sizeof(float))))
        return AVERROR(ENOMEM);

    if (s->tonemap == TONEMAP_BT2390)
        peak = inverse_eotf_st2084(peak);

    for (i = 0; i < 32768; i++) {
        double v = (i - 2048.0) / 28672.0;
        double lin = eotf_st2084(v);
        double mapval = s->tonemap == TONEMAP_BT2390 ? v : lin;
        float mapped = mapsig(s->tonemap, mapval, peak, s->param);
        s->tonemap_lut[i] = (lin > 0 && mapped > 0) ? mapped / lin : 0;
    }

    return 0;
}

static int compute_yuv_coeffs(TonemapContext *s,
                              const struct LumaCoefficients *coeffs,
                              const struct LumaCoefficients *ocoeffs,
                              const AVPixFmtDescriptor *idesc,
                              const AVPixFmtDescriptor *odesc,
                              enum AVColorRange irng,
                              enum AVColorRange orng)
{
    double rgb2yuv[3][3], yuv2rgb[3][3];
    int res;
    int y_rng, uv_rng;

    res = ff_get_range_off(&s->in_yuv_off, &y_rng, &uv_rng,
                           irng, idesc->comp[0].depth);
    if (res < 0) {
        av_log(s, AV_LOG_ERROR,
               "Unsupported input color range %d (%s)\n",
               irng, av_color_range_name(irng));
        return res;
    }

    ff_fill_rgb2yuv_table(coeffs, rgb2yuv);
    ff_matrix_invert_3x3(rgb2yuv, yuv2rgb);
    ff_fill_rgb2yuv_table(ocoeffs, rgb2yuv);

    ff_get_yuv_coeffs(s->yuv2rgb_coeffs, yuv2rgb, idesc->comp[0].depth,
                      y_rng, uv_rng, 1);

    res = ff_get_range_off(&s->out_yuv_off, &y_rng, &uv_rng,
                           orng, odesc->comp[0].depth);
    if (res < 0) {
        av_log(s, AV_LOG_ERROR,
               "Unsupported output color range %d (%s)\n",
               orng, av_color_range_name(orng));
        return res;
    }

    ff_get_yuv_coeffs(s->rgb2yuv_coeffs, rgb2yuv, odesc->comp[0].depth,
                      y_rng, uv_rng, 0);

    return 0;
}


static int compute_rgb_coeffs(TonemapContext *s,
                              enum AVColorPrimaries iprm,
                              enum AVColorPrimaries oprm)
{
    double rgb2xyz[3][3], xyz2rgb[3][3];
    const struct WhitepointCoefficients wp = { 0.3127, 0.3290 };
    const struct PrimaryCoefficients *icoeff = ff_get_color_primaries(iprm);
    const struct PrimaryCoefficients *ocoeff = ff_get_color_primaries(oprm);

    if (!icoeff) {
        av_log(s, AV_LOG_ERROR,
               "Unsupported input color primaries %d (%s)\n",
               iprm, av_color_primaries_name(iprm));
        return AVERROR(EINVAL);
    }
    if (!ocoeff) {
        av_log(s, AV_LOG_ERROR,
               "Unsupported output color primaries %d (%s)\n",
               oprm, av_color_primaries_name(oprm));
        return AVERROR(EINVAL);
    }

    ff_fill_rgb2xyz_table(ocoeff, &wp, rgb2xyz);
    ff_matrix_invert_3x3(rgb2xyz, xyz2rgb);
    ff_fill_rgb2xyz_table(icoeff, &wp, rgb2xyz);
    ff_matrix_mul_3x3(s->rgb2rgb_coeffs, rgb2xyz, xyz2rgb);

    return 0;
}

static int filter_slice(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)
{
    TonemapContext *s = ctx->priv;
    ThreadData *td = arg;
    AVFrame *in = td->in;
    AVFrame *out = td->out;
    const AVPixFmtDescriptor *desc  = td->desc;
    const AVPixFmtDescriptor *odesc = td->odesc;
    const int ss = 1 << FFMAX(desc->log2_chroma_h, odesc->log2_chroma_h);
    const int slice_start = (in->height / ss *  jobnr   ) / nb_jobs * ss;
    const int slice_end   = (in->height / ss * (jobnr+1)) / nb_jobs * ss;
    int y, x;

    if (desc->flags & AV_PIX_FMT_FLAG_FLOAT) {
        /* do the tone map */
#define COMP(frame, c) ((float*)(frame->data[c] + x * desc->comp[c].step + y * frame->linesize[c]))
        for (y = slice_start; y < slice_end; y++) {
            for (x = 0; x < out->width; x++) {
                tonemap(*COMP(in, 0), *COMP(in, 1), *COMP(in, 2),
                        COMP(out, 0), COMP(out, 1), COMP(out, 2),
                        s->param, s->desat, td->peak,
                        s->coeffs, s->tonemap);
            }
        }
    } else {
        TonemapIntParams params = {
            .lut_peak       = s->lut_peak,
            .lin_lut        = s->lin_lut,
            .tonemap_lut    = s->tonemap_lut,
            .delin_lut      = s->delin_lut,
            .in_yuv_off     = s->in_yuv_off,
            .out_yuv_off    = s->out_yuv_off,
            .yuv2rgb_coeffs = &s->yuv2rgb_coeffs,
            .rgb2yuv_coeffs = &s->rgb2yuv_coeffs,
            .rgb2rgb_coeffs = &s->rgb2rgb_coeffs,
            .coeffs         = s->coeffs,
            .ocoeffs        = s->ocoeffs,
            .desat          = s->desat,
        };
        s->tonemap_frame_p010_nv12(out->data[0] + out->linesize[0] * slice_start,
                                   out->data[1] + out->linesize[1] * AV_CEIL_RSHIFT(slice_start, desc->log2_chroma_h),
                                   (void*)(in->data[0] + in->linesize[0] * slice_start),
                                   (void*)(in->data[1] + in->linesize[1] * AV_CEIL_RSHIFT(slice_start, odesc->log2_chroma_h)),
                                   out->linesize, in->linesize, out->width,
                                   slice_end - slice_start,
                                   &params);
    }

    /* copy/generate alpha if needed */
    if (desc->flags & AV_PIX_FMT_FLAG_ALPHA && odesc->flags & AV_PIX_FMT_FLAG_ALPHA) {
        av_image_copy_plane(out->data[3] + out->linesize[3] * slice_start, out->linesize[3],
                            in->data[3] + in->linesize[3] * slice_start, in->linesize[3],
                            out->linesize[3], slice_end - slice_start);
    } else if (odesc->flags & AV_PIX_FMT_FLAG_ALPHA) {
        for (y = slice_start; y < slice_end; y++) {
            for (x = 0; x < out->width; x++) {
                AV_WN32(out->data[3] + x * odesc->comp[3].step + y * out->linesize[3],
                        av_float2int(1.0f));
            }
        }
    }

    return 0;
}

static int filter_frame(AVFilterLink *link, AVFrame *in)
{
    AVFilterContext *ctx = link->dst;
    TonemapContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    ThreadData td;
    AVFrame *out;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(link->format);
    const AVPixFmtDescriptor *odesc = av_pix_fmt_desc_get(outlink->format);
    int ret;
    double peak = s->peak;
    const struct LumaCoefficients *coeffs = ff_get_luma_coefficients(in->colorspace);

    if (!desc || !odesc) {
        av_frame_free(&in);
        return AVERROR_BUG;
    }

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }

    if ((ret = av_frame_copy_props(out, in)) < 0)
        goto fail;

    /* read peak from side data if not passed in */
    if (!peak) {
        peak = ff_determine_signal_peak(in);
        av_log(s, AV_LOG_DEBUG, "Computed signal peak: %f\n", peak);
    }

    /* input and output transfer will be linear */
    if (desc->flags & AV_PIX_FMT_FLAG_FLOAT) {
        if (in->color_trc == AVCOL_TRC_UNSPECIFIED) {
            av_log(s, AV_LOG_WARNING, "Untagged transfer, assuming linear light\n");
            out->color_trc = AVCOL_TRC_LINEAR;
        } else if (in->color_trc != AVCOL_TRC_LINEAR)
            av_log(s, AV_LOG_WARNING, "Tonemapping works on linear light only\n");
    } else {
        if (in->color_trc == AVCOL_TRC_UNSPECIFIED) {
            av_log(s, AV_LOG_WARNING, "Untagged transfer, assuming PQ\n");
            out->color_trc = AVCOL_TRC_SMPTEST2084;
        } else if (in->color_trc != AVCOL_TRC_SMPTEST2084)
            av_log(s, AV_LOG_WARNING, "Tonemapping works on PQ only\n");

        out->color_trc       = AVCOL_TRC_BT709;
        out->colorspace      = AVCOL_SPC_BT709;
        out->color_primaries = AVCOL_PRI_BT709;

        if (!s->lin_lut || !s->delin_lut) {
            if ((ret = comput_trc_luts(s, in->color_trc, out->color_trc)) < 0)
                goto fail;
        }

        if (!s->tonemap_lut || s->lut_peak != peak) {
            s->lut_peak = peak;
            if ((ret = compute_tonemap_lut(s)) < 0)
                goto fail;
        }

        if (s->coeffs != coeffs) {
            enum AVColorPrimaries iprm = in->color_primaries;
            s->ocoeffs = ff_get_luma_coefficients(out->colorspace);
            if ((ret = compute_yuv_coeffs(s, coeffs, s->ocoeffs, desc, odesc,
                 in->color_range, out->color_range)) < 0)
                goto fail;
            if (iprm == AVCOL_PRI_UNSPECIFIED)
                iprm = AVCOL_PRI_BT2020;
            if ((ret = compute_rgb_coeffs(s, iprm, out->color_primaries)) < 0)
                goto fail;
        }
    }

    /* load original color space even if pixel format is RGB to compute overbrights */
    s->coeffs = coeffs;
    if (s->desat > 0 && !s->coeffs) {
        if (in->colorspace == AVCOL_SPC_UNSPECIFIED)
            av_log(s, AV_LOG_WARNING, "Missing color space information, ");
        else if (!s->coeffs)
            av_log(s, AV_LOG_WARNING, "Unsupported color space '%s', ",
                   av_color_space_name(in->colorspace));
        av_log(s, AV_LOG_WARNING, "desaturation is disabled\n");
        s->desat = 0;
    }

    td.in    = in;
    td.out   = out;
    td.desc  = desc;
    td.odesc = odesc;
    td.peak  = peak;
    ctx->internal->execute(ctx, filter_slice, &td, NULL, FFMIN(outlink->h >> FFMAX(desc->log2_chroma_h, odesc->log2_chroma_h), ctx->graph->nb_threads));

    av_frame_free(&in);

    ff_update_hdr_metadata(out, peak);

    return ff_filter_frame(outlink, out);
fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

static void uninit(AVFilterContext *ctx)
{
    TonemapContext *s = ctx->priv;

    av_freep(&s->lin_lut);
    av_freep(&s->delin_lut);
    av_freep(&s->tonemap_lut);
}

#define OFFSET(x) offsetof(TonemapContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM
static const AVOption tonemap_options[] = {
    { "tonemap",      "tonemap algorithm selection", OFFSET(tonemap), AV_OPT_TYPE_INT, {.i64 = TONEMAP_BT2390}, TONEMAP_NONE, TONEMAP_MAX - 1, FLAGS, "tonemap" },
    {     "none",     0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_NONE},              0, 0, FLAGS, "tonemap" },
    {     "linear",   0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_LINEAR},            0, 0, FLAGS, "tonemap" },
    {     "gamma",    0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_GAMMA},             0, 0, FLAGS, "tonemap" },
    {     "clip",     0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_CLIP},              0, 0, FLAGS, "tonemap" },
    {     "reinhard", 0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_REINHARD},          0, 0, FLAGS, "tonemap" },
    {     "hable",    0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_HABLE},             0, 0, FLAGS, "tonemap" },
    {     "mobius",   0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_MOBIUS},            0, 0, FLAGS, "tonemap" },
    {     "bt2390",   0, 0, AV_OPT_TYPE_CONST, {.i64 = TONEMAP_BT2390},            0, 0, FLAGS, "tonemap" },
    { "param",        "tonemap parameter", OFFSET(param), AV_OPT_TYPE_DOUBLE, {.dbl = NAN}, DBL_MIN, DBL_MAX, FLAGS },
    { "desat",        "desaturation strength", OFFSET(desat), AV_OPT_TYPE_DOUBLE, {.dbl = 2}, 0, DBL_MAX, FLAGS },
    { "peak",         "signal peak override", OFFSET(peak), AV_OPT_TYPE_DOUBLE, {.dbl = 0}, 0, DBL_MAX, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(tonemap);

static const AVFilterPad tonemap_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },
    { NULL }
};

static const AVFilterPad tonemap_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

AVFilter ff_vf_tonemap = {
    .name            = "tonemap",
    .description     = NULL_IF_CONFIG_SMALL("Conversion to/from different dynamic ranges."),
    .init            = init,
    .uninit          = uninit,
    .query_formats   = query_formats,
    .priv_size       = sizeof(TonemapContext),
    .priv_class      = &tonemap_class,
    .inputs          = tonemap_inputs,
    .outputs         = tonemap_outputs,
    .flags           = AVFILTER_FLAG_SLICE_THREADS,
};
