/*
 * Copyright (c) 2020 rcombs
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "colorspace_common.h"
#include "pixfmt.h"
#include "tonemap.h"
#include "util.h"

extern const enum TonemapAlgorithm tonemap_func;
extern const float tone_param;

#define mix(x, y, a) ((x) + ((y) - (x)) * (a))

static __inline__ __device__
float hable_f(float in) {
    float a = 0.15f, b = 0.50f, c = 0.10f, d = 0.20f, e = 0.02f, f = 0.30f;
    return (in * (in * a + b * c) + d * e) / (in * (in * a + b) + d * f) - e / f;
}

static __inline__ __device__
float direct(float s, float peak) {
    return s;
}

static __inline__ __device__
float linear(float s, float peak) {
    return s * tone_param / peak;
}

static __inline__ __device__
float gamma(float s, float peak) {
    float p = s > 0.05f ? s / peak : 0.05f / peak;
    float v = __powf(p, 1.0f / tone_param);
    return s > 0.05f ? v : (s * v /0.05f);
}

static __inline__ __device__
float clip(float s, float peak) {
    return clamp(s * tone_param, 0.0f, 1.0f);
}

static __inline__ __device__
float reinhard(float s, float peak) {
    return s / (s + tone_param) * (peak + tone_param) / peak;
}

static __inline__ __device__
float hable(float s, float peak) {
    return hable_f(s) / hable_f(peak);
}

static __inline__ __device__
float mobius(float s, float peak) {
    float j = tone_param;
    float a, b;

    if (s <= j)
        return s;

    a = -j * j * (peak - 1.0f) / (j * j - 2.0f * j + peak);
    b = (j * j - 2.0f * j * peak + peak) / max(peak - 1.0f, 1e-6f);

    return (b * b + 2.0f * b * j + j * j) / (b - a) * (s + a) / (s + b);
}

static __inline__ __device__
float map(float s, float peak)
{
    switch (tonemap_func) {
    case TONEMAP_NONE:
    default:
        return direct(s, peak);
    case TONEMAP_LINEAR:
        return linear(s, peak);
    case TONEMAP_GAMMA:
        return gamma(s, peak);
    case TONEMAP_CLIP:
        return clip(s, peak);
    case TONEMAP_REINHARD:
        return reinhard(s, peak);
    case TONEMAP_HABLE:
        return hable(s, peak);
    case TONEMAP_MOBIUS:
        return mobius(s, peak);
    }
}

static __inline__ __device__
float3 map_one_pixel_rgb(float3 rgb, const FFCUDAFrame& src, const FFCUDAFrame& dst) {
    float sig = max(max(rgb.x, max(rgb.y, rgb.z)), 1e-6f);
    float peak = src.peak;

    // Rescale the variables in order to bring it into a representation where
    // 1.0 represents the dst_peak. This is because all of the tone mapping
    // algorithms are defined in such a way that they map to the range [0.0, 1.0].
    if (dst.peak > 1.0f) {
        sig *= 1.0f / dst.peak;
        peak *= 1.0f / dst.peak;
    }

    float sig_old = sig;

    /*
    // Scale the signal to compensate for differences in the average brightness
    float slope = min(1.0f, dst.average / src.average);
    sig *= slope;
    peak *= slope;
    */

    // Desaturate the color using a coefficient dependent on the signal level
    /*
    if (desat_param > 0.0f) {
        float luma = dstSpace.getLuma(rgb);
        float coeff = max(sig - 0.18f, 1e-6f) / max(sig, 1e-6f);
        coeff = __powf(coeff, 10.0f / desat_param);
        rgb = mix(rgb, (float3)luma, (float3)coeff);
        sig = mix(sig, luma * slope, coeff);
    }
    */

    sig = map(sig, peak);

    sig = min(sig, 1.0f);
    rgb = rgb * (sig/sig_old);
    return rgb;
}

// map from source space YUV to destination space RGB
static __inline__ __device__
float3 map_to_dst_space_from_yuv(float3 yuv, float peak) {
    float3 c = yuv2lrgb(yuv);
    c = ootf(c, peak);
    c = lrgb2lrgb(c);
    return c;
}


extern "C" {

__global__ void tonemap(FFCUDAFrame src, FFCUDAFrame dst)
{
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    // each work item process four pixels
    int x = 2 * xi;
    int y = 2 * yi;

    if (y + 1 < src.height && x + 1 < src.width)
    {
        float3 yuv0 = read_px_flt(src, x,     y);
        float3 yuv1 = read_px_flt(src, x + 1, y);
        float3 yuv2 = read_px_flt(src, x,     y + 1);
        float3 yuv3 = read_px_flt(src, x + 1, y + 1);

        float3 c0 = map_to_dst_space_from_yuv(yuv0, src.peak);
        float3 c1 = map_to_dst_space_from_yuv(yuv1, src.peak);
        float3 c2 = map_to_dst_space_from_yuv(yuv2, src.peak);
        float3 c3 = map_to_dst_space_from_yuv(yuv3, src.peak);

        c0 = map_one_pixel_rgb(c0, src, dst);
        c1 = map_one_pixel_rgb(c1, src, dst);
        c2 = map_one_pixel_rgb(c2, src, dst);
        c3 = map_one_pixel_rgb(c3, src, dst);

        c0 = inverse_ootf(c0, dst.peak);
        c1 = inverse_ootf(c1, dst.peak);
        c2 = inverse_ootf(c2, dst.peak);
        c3 = inverse_ootf(c3, dst.peak);

        yuv0 = lrgb2yuv(c0);
        yuv1 = lrgb2yuv(c1);
        yuv2 = lrgb2yuv(c2);
        yuv3 = lrgb2yuv(c3);

        write_2x2_flt(dst, x, y, yuv0, yuv1, yuv2, yuv3);
    }
}

}
