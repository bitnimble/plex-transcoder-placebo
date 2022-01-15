/*
 * Copyright (c) 2016 Ronald S. Bultje <rsbultje@gmail.com>
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

#include "libavutil/avassert.h"
#include "libavutil/frame.h"
#include "libavutil/mastering_display_metadata.h"
#include "libavutil/pixdesc.h"

#include "colorspace.h"


void ff_matrix_invert_3x3(const double in[3][3], double out[3][3])
{
    double m00 = in[0][0], m01 = in[0][1], m02 = in[0][2],
           m10 = in[1][0], m11 = in[1][1], m12 = in[1][2],
           m20 = in[2][0], m21 = in[2][1], m22 = in[2][2];
    int i, j;
    double det;

    out[0][0] =  (m11 * m22 - m21 * m12);
    out[0][1] = -(m01 * m22 - m21 * m02);
    out[0][2] =  (m01 * m12 - m11 * m02);
    out[1][0] = -(m10 * m22 - m20 * m12);
    out[1][1] =  (m00 * m22 - m20 * m02);
    out[1][2] = -(m00 * m12 - m10 * m02);
    out[2][0] =  (m10 * m21 - m20 * m11);
    out[2][1] = -(m00 * m21 - m20 * m01);
    out[2][2] =  (m00 * m11 - m10 * m01);

    det = m00 * out[0][0] + m10 * out[0][1] + m20 * out[0][2];
    det = 1.0 / det;

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++)
            out[i][j] *= det;
    }
}

void ff_matrix_mul_3x3(double dst[3][3],
               const double src1[3][3], const double src2[3][3])
{
    int m, n;

    for (m = 0; m < 3; m++)
        for (n = 0; n < 3; n++)
            dst[m][n] = src2[m][0] * src1[0][n] +
                        src2[m][1] * src1[1][n] +
                        src2[m][2] * src1[2][n];
}
/*
 * see e.g. http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
 */
void ff_fill_rgb2xyz_table(const struct PrimaryCoefficients *coeffs,
                           const struct WhitepointCoefficients *wp,
                           double rgb2xyz[3][3])
{
    double i[3][3], sr, sg, sb, zw;

    rgb2xyz[0][0] = coeffs->xr / coeffs->yr;
    rgb2xyz[0][1] = coeffs->xg / coeffs->yg;
    rgb2xyz[0][2] = coeffs->xb / coeffs->yb;
    rgb2xyz[1][0] = rgb2xyz[1][1] = rgb2xyz[1][2] = 1.0;
    rgb2xyz[2][0] = (1.0 - coeffs->xr - coeffs->yr) / coeffs->yr;
    rgb2xyz[2][1] = (1.0 - coeffs->xg - coeffs->yg) / coeffs->yg;
    rgb2xyz[2][2] = (1.0 - coeffs->xb - coeffs->yb) / coeffs->yb;
    ff_matrix_invert_3x3(rgb2xyz, i);
    zw = 1.0 - wp->xw - wp->yw;
    sr = i[0][0] * wp->xw + i[0][1] * wp->yw + i[0][2] * zw;
    sg = i[1][0] * wp->xw + i[1][1] * wp->yw + i[1][2] * zw;
    sb = i[2][0] * wp->xw + i[2][1] * wp->yw + i[2][2] * zw;
    rgb2xyz[0][0] *= sr;
    rgb2xyz[0][1] *= sg;
    rgb2xyz[0][2] *= sb;
    rgb2xyz[1][0] *= sr;
    rgb2xyz[1][1] *= sg;
    rgb2xyz[1][2] *= sb;
    rgb2xyz[2][0] *= sr;
    rgb2xyz[2][1] *= sg;
    rgb2xyz[2][2] *= sb;
}
static const double ycgco_matrix[3][3] =
{
    {  0.25, 0.5,  0.25 },
    { -0.25, 0.5, -0.25 },
    {  0.5,  0,   -0.5  },
};

static const double gbr_matrix[3][3] =
{
    { 0,    1,   0   },
    { 0,   -0.5, 0.5 },
    { 0.5, -0.5, 0   },
};

/*
 * All constants explained in e.g. https://linuxtv.org/downloads/v4l-dvb-apis/ch02s06.html
 * The older ones (bt470bg/m) are also explained in their respective ITU docs
 * (e.g. https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf)
 * whereas the newer ones can typically be copied directly from wikipedia :)
 */
static const struct LumaCoefficients luma_coefficients[AVCOL_SPC_NB] = {
    [AVCOL_SPC_FCC]        = { 0.30,   0.59,   0.11   },
    [AVCOL_SPC_BT470BG]    = { 0.299,  0.587,  0.114  },
    [AVCOL_SPC_SMPTE170M]  = { 0.299,  0.587,  0.114  },
    [AVCOL_SPC_BT709]      = { 0.2126, 0.7152, 0.0722 },
    [AVCOL_SPC_SMPTE240M]  = { 0.212,  0.701,  0.087  },
    [AVCOL_SPC_YCOCG]      = { 0.25,   0.5,    0.25   },
    [AVCOL_SPC_RGB]        = { 1,      1,      1      },
    [AVCOL_SPC_BT2020_NCL] = { 0.2627, 0.6780, 0.0593 },
    [AVCOL_SPC_BT2020_CL]  = { 0.2627, 0.6780, 0.0593 },
};

const struct LumaCoefficients *ff_get_luma_coefficients(enum AVColorSpace csp)
{
    const struct LumaCoefficients *coeffs;

    if (csp >= AVCOL_SPC_NB)
        return NULL;
    coeffs = &luma_coefficients[csp];
    if (!coeffs->cr)
        return NULL;

    return coeffs;
}

static const struct PrimaryCoefficients color_primaries[AVCOL_PRI_NB] = {
    [AVCOL_PRI_BT709]     = { 0.640, 0.330, 0.300, 0.600, 0.150, 0.060 },
    [AVCOL_PRI_BT470M]    = { 0.670, 0.330, 0.210, 0.710, 0.140, 0.080 },
    [AVCOL_PRI_BT470BG]   = { 0.640, 0.330, 0.290, 0.600, 0.150, 0.060 },
    [AVCOL_PRI_SMPTE170M] = { 0.630, 0.340, 0.310, 0.595, 0.155, 0.070 },
    [AVCOL_PRI_SMPTE240M] = { 0.630, 0.340, 0.310, 0.595, 0.155, 0.070 },
    [AVCOL_PRI_SMPTE428]  = { 0.735, 0.265, 0.274, 0.718, 0.167, 0.009 },
    [AVCOL_PRI_SMPTE431]  = { 0.680, 0.320, 0.265, 0.690, 0.150, 0.060 },
    [AVCOL_PRI_SMPTE432]  = { 0.680, 0.320, 0.265, 0.690, 0.150, 0.060 },
    [AVCOL_PRI_FILM]      = { 0.681, 0.319, 0.243, 0.692, 0.145, 0.049 },
    [AVCOL_PRI_BT2020]    = { 0.708, 0.292, 0.170, 0.797, 0.131, 0.046 },
    [AVCOL_PRI_JEDEC_P22] = { 0.630, 0.340, 0.295, 0.605, 0.155, 0.077 },
};

const struct PrimaryCoefficients *ff_get_color_primaries(enum AVColorPrimaries prm)
{
    const struct PrimaryCoefficients *p;

    if (prm >= AVCOL_PRI_NB)
        return NULL;
    p = &color_primaries[prm];
    if (!p->xr)
        return NULL;

    return p;
}

void ff_fill_rgb2yuv_table(const struct LumaCoefficients *coeffs,
                           double rgb2yuv[3][3])
{
    double bscale, rscale;

    // special ycgco matrix
    if (coeffs->cr == 0.25 && coeffs->cg == 0.5 && coeffs->cb == 0.25) {
        memcpy(rgb2yuv, ycgco_matrix, sizeof(double) * 9);
        return;
    } else if (coeffs->cr == 1 && coeffs->cg == 1 && coeffs->cb == 1) {
        memcpy(rgb2yuv, gbr_matrix, sizeof(double) * 9);
        return;
    }

    rgb2yuv[0][0] = coeffs->cr;
    rgb2yuv[0][1] = coeffs->cg;
    rgb2yuv[0][2] = coeffs->cb;
    bscale = 0.5 / (coeffs->cb - 1.0);
    rscale = 0.5 / (coeffs->cr - 1.0);
    rgb2yuv[1][0] = bscale * coeffs->cr;
    rgb2yuv[1][1] = bscale * coeffs->cg;
    rgb2yuv[1][2] = 0.5;
    rgb2yuv[2][0] = 0.5;
    rgb2yuv[2][1] = rscale * coeffs->cg;
    rgb2yuv[2][2] = rscale * coeffs->cb;
}

int ff_get_range_off(int *off, int *y_rng, int *uv_rng,
                     enum AVColorRange rng, int depth)
{
    switch (rng) {
    case AVCOL_RANGE_UNSPECIFIED:
    case AVCOL_RANGE_MPEG:
        *off = 16 << (depth - 8);
        *y_rng = 219 << (depth - 8);
        *uv_rng = 224 << (depth - 8);
        break;
    case AVCOL_RANGE_JPEG:
        *off = 0;
        *y_rng = *uv_rng = (256 << (depth - 8)) - 1;
        break;
    default:
        return AVERROR(EINVAL);
    }

    return 0;
}

void ff_get_yuv_coeffs(int16_t out[3][3][8], double (*table)[3],
                       int depth, int y_rng, int uv_rng, int yuv2rgb)
{
#define N (yuv2rgb ? m : n)
#define M (yuv2rgb ? n : m)
    int rng, n, m, o;
    int bits = 1 << (yuv2rgb ? (depth - 1) : (29 - depth));
    for (rng = y_rng, n = 0; n < 3; n++, rng = uv_rng) {
        for (m = 0; m < 3; m++) {
            out[N][M][0] = lrint(bits * (yuv2rgb ? 28672 : rng) * table[N][M] / (yuv2rgb ? rng : 28672));
            for (o = 1; o < 8; o++)
                out[N][M][o] = out[N][M][0];
        }
    }

    if (yuv2rgb) {
        av_assert2(out[0][1][0] == 0);
        av_assert2(out[2][2][0] == 0);
        av_assert2(out[0][0][0] == out[1][0][0]);
        av_assert2(out[0][0][0] == out[2][0][0]);
    } else {
        av_assert2(out[1][2][0] == out[2][0][0]);
    }
}

double ff_determine_signal_peak(const AVFrame *in)
{
    AVFrameSideData *sd = av_frame_get_side_data(in, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
    double peak = 0;

    if (sd) {
        AVContentLightMetadata *clm = (AVContentLightMetadata *)sd->data;
        peak = clm->MaxCLL / REFERENCE_WHITE;
    }

    sd = av_frame_get_side_data(in, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
    if (!peak && sd) {
        AVMasteringDisplayMetadata *metadata = (AVMasteringDisplayMetadata *)sd->data;
        if (metadata->has_luminance)
            peak = av_q2d(metadata->max_luminance) / REFERENCE_WHITE;
    }

    // For untagged source, use peak of 10000 if SMPTE ST.2084
    // otherwise assume HLG with reference display peak 1000.
    if (!peak)
        peak = in->color_trc == AVCOL_TRC_SMPTE2084 ? 100.0f : 10.0f;

    return peak;
}

void ff_update_hdr_metadata(AVFrame *in, double peak)
{
    AVFrameSideData *sd = av_frame_get_side_data(in, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);

    if (sd) {
        AVContentLightMetadata *clm = (AVContentLightMetadata *)sd->data;
        clm->MaxCLL = (unsigned)(peak * REFERENCE_WHITE);
    }

    sd = av_frame_get_side_data(in, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
    if (sd) {
        AVMasteringDisplayMetadata *metadata = (AVMasteringDisplayMetadata *)sd->data;
        if (metadata->has_luminance)
            metadata->max_luminance = av_d2q(peak * REFERENCE_WHITE, 10000);
    }
}
