/*
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

#ifndef AVFILTER_CUDA_COLORSPACE_COMMON_H
#define AVFILTER_CUDA_COLORSPACE_COMMON_H

#include "util.h"
#include "libavutil/pixfmt.h"

#define ST2084_MAX_LUMINANCE 10000.0f
#define REFERENCE_WHITE 100.0f

#define ST2084_M1 0.1593017578125f
#define ST2084_M2 78.84375f
#define ST2084_C1 0.8359375f
#define ST2084_C2 18.8515625f
#define ST2084_C3 18.6875f
#define SDR_AVG   0.25f

extern const float3 luma_src, luma_dst;
extern const enum AVColorTransferCharacteristic trc_src, trc_dst;
extern const enum AVColorRange range_src, range_dst;
extern const enum AVChromaLocation chroma_loc_src, chroma_loc_dst;
extern const bool rgb2rgb_passthrough;
extern const float rgb2rgb_matrix[9];
extern const float yuv_matrix[9], rgb_matrix[9];

static __inline__ __device__ float get_luma_dst(float3 c, const float3& luma_dst) {
    return luma_dst.x * c.x + luma_dst.y * c.y + luma_dst.z * c.z;
}

static __inline__ __device__ float get_luma_src(float3 c, const float3& luma_src) {
    return luma_src.x * c.x + luma_src.y * c.y + luma_src.z * c.z;
}

static __inline__ __device__ float3 get_chroma_sample(float3 a, float3 b, float3 c, float3 d) {
    switch (chroma_loc_dst) {
    case AVCHROMA_LOC_LEFT:
        return ((a) + (c)) * 0.5f;
    case AVCHROMA_LOC_CENTER:
    case AVCHROMA_LOC_UNSPECIFIED:
    default:
        return ((a) + (b) + (c) + (d)) * 0.25f;
    case AVCHROMA_LOC_TOPLEFT:
        return a;
    case AVCHROMA_LOC_TOP:
        return ((a) + (b)) * 0.5f;
    case AVCHROMA_LOC_BOTTOMLEFT:
        return c;
    case AVCHROMA_LOC_BOTTOM:
        return ((c) + (d)) * 0.5f;
    }
}

static __inline__ __device__ float eotf_st2084(float x) {
    float p = __powf(x, 1.0f / ST2084_M2);
    float a = max(p -ST2084_C1, 0.0f);
    float b = max(ST2084_C2 - ST2084_C3 * p, 1e-6f);
    float c  = __powf(a / b, 1.0f / ST2084_M1);
    return x > 0.0f ? c * ST2084_MAX_LUMINANCE / REFERENCE_WHITE : 0.0f;
}

#define HLG_A 0.17883277f
#define HLG_B 0.28466892f
#define HLG_C 0.55991073f

// linearizer for HLG
static __inline__ __device__ float inverse_oetf_hlg(float x) {
    float a = 4.0f * x * x;
    float b = __expf((x - HLG_C) / HLG_A) + HLG_B;
    return x < 0.5f ? a : b;
}

// delinearizer for HLG
static __inline__ __device__ float oetf_hlg(float x) {
    float a = 0.5f * __sqrtf(x);
    float b = HLG_A * __logf(x - HLG_B) + HLG_C;
    return x <= 1.0f ? a : b;
}

static __inline__ __device__ float3 ootf_hlg(float3 c, float peak) {
    float luma = get_luma_src(c, luma_src);
    float gamma =  1.2f + 0.42f * __log10f(peak * REFERENCE_WHITE / 1000.0f);
    gamma = max(1.0f, gamma);
    float factor = peak * __powf(luma, gamma - 1.0f) / __powf(12.0f, gamma);
    return c * factor;
}

static __inline__ __device__ float3 inverse_ootf_hlg(float3 c, float peak) {
    float gamma = 1.2f + 0.42f * __log10f(peak * REFERENCE_WHITE / 1000.0f);
    c = c * __powf(12.0f, gamma) / peak;
    c = c / __powf(get_luma_dst(c, luma_dst), (gamma - 1.0f) / gamma);
    return c;
}

static __inline__ __device__ float inverse_eotf_bt1886(float c) {
    return c < 0.0f ? 0.0f : __powf(c, 1.0f / 2.4f);
}

static __inline__ __device__ float oetf_bt709(float c) {
    c = c < 0.0f ? 0.0f : c;
    float r1 = 4.5f * c;
    float r2 = 1.099f * __powf(c, 0.45f) - 0.099f;
    return c < 0.018f ? r1 : r2;
}
static __inline__ __device__ float inverse_oetf_bt709(float c) {
    float r1 = c / 4.5f;
    float r2 = __powf((c + 0.099f) / 1.099f, 1.0f / 0.45f);
    return c < 0.081f ? r1 : r2;
}

static __inline__ __device__ float3 ootf(float3 c, float peak)
{
    if (trc_src == AVCOL_TRC_ARIB_STD_B67)
        return ootf_hlg(c, peak);
    else
        return c;
}

static __inline__ __device__ float3 inverse_ootf(float3 c, float peak)
{
    if (trc_dst == AVCOL_TRC_ARIB_STD_B67)
        return inverse_ootf_hlg(c, peak);
    else
        return c;
}

static __inline__ __device__ float linearize(float x)
{
    if (trc_src == AVCOL_TRC_SMPTE2084)
        return eotf_st2084(x);
    else if (trc_src == AVCOL_TRC_ARIB_STD_B67)
        return inverse_oetf_hlg(x);
    else
        return x;
}

static __inline__ __device__ float delinearize(float x)
{
    if (trc_dst == AVCOL_TRC_BT709 || trc_dst == AVCOL_TRC_BT2020_10)
        return inverse_eotf_bt1886(x);
    else
        return x;
}

static __inline__ __device__ float3 yuv2rgb(float y, float u, float v) {
    if (range_src == AVCOL_RANGE_JPEG) {
        u -= 0.5f; v -= 0.5f;
    } else {
        y = (y * 255.0f -  16.0f) / 219.0f;
        u = (u * 255.0f - 128.0f) / 224.0f;
        v = (v * 255.0f - 128.0f) / 224.0f;
    }
    float r = y * rgb_matrix[0] + u * rgb_matrix[1] + v * rgb_matrix[2];
    float g = y * rgb_matrix[3] + u * rgb_matrix[4] + v * rgb_matrix[5];
    float b = y * rgb_matrix[6] + u * rgb_matrix[7] + v * rgb_matrix[8];
    return make_float3(r, g, b);
}

static __inline__ __device__ float3 yuv2lrgb(float3 yuv) {
    float3 rgb = yuv2rgb(yuv.x, yuv.y, yuv.z);
    return make_float3(linearize(rgb.x),
                       linearize(rgb.y),
                       linearize(rgb.z));
}

static __inline__ __device__ float3 rgb2yuv(float r, float g, float b) {
    float y = r*yuv_matrix[0] + g*yuv_matrix[1] + b*yuv_matrix[2];
    float u = r*yuv_matrix[3] + g*yuv_matrix[4] + b*yuv_matrix[5];
    float v = r*yuv_matrix[6] + g*yuv_matrix[7] + b*yuv_matrix[8];
    if (range_dst == AVCOL_RANGE_JPEG) {
        u += 0.5f; v += 0.5f;
    } else {
        y = (219.0f * y + 16.0f) / 255.0f;
        u = (224.0f * u + 128.0f) / 255.0f;
        v = (224.0f * v + 128.0f) / 255.0f;
    }
    return make_float3(y, u, v);
}

static __inline__ __device__ float rgb2y(float r, float g, float b) {
    float y = r*yuv_matrix[0] + g*yuv_matrix[1] + b*yuv_matrix[2];
    if (range_dst != AVCOL_RANGE_JPEG)
        y = (219.0f * y + 16.0f) / 255.0f;
    return y;
}

static __inline__ __device__ float3 lrgb2yuv(float3 c) {
    float r = delinearize(c.x);
    float g = delinearize(c.y);
    float b = delinearize(c.z);
    return rgb2yuv(r, g, b);
}

static __inline__ __device__ float3 lrgb2lrgb(float3 c) {
    if (rgb2rgb_passthrough) {
        return c;
    } else {
        float r = c.x, g = c.y, b = c.z;
        float rr = rgb2rgb_matrix[0] * r + rgb2rgb_matrix[1] * g + rgb2rgb_matrix[2] * b;
        float gg = rgb2rgb_matrix[3] * r + rgb2rgb_matrix[4] * g + rgb2rgb_matrix[5] * b;
        float bb = rgb2rgb_matrix[6] * r + rgb2rgb_matrix[7] * g + rgb2rgb_matrix[8] * b;
        return make_float3(rr, gg, bb);
    }
}

#endif /* AVFILTER_CUDA_COLORSPACE_COMMON_H */
