/*
 * Copyright (c) 2020 rcombs <rcombs@rcombs.me>
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

#ifndef AVFILTER_TONEMAP_H
#define AVFILTER_TONEMAP_H

#include "avfilter.h"
#include "colorspace.h"

enum TonemapAlgorithm {
    TONEMAP_NONE,
    TONEMAP_LINEAR,
    TONEMAP_GAMMA,
    TONEMAP_CLIP,
    TONEMAP_REINHARD,
    TONEMAP_HABLE,
    TONEMAP_MOBIUS,
    TONEMAP_BT2390,
    TONEMAP_MAX,
};

typedef struct TonemapIntParams {
    double lut_peak;
    float *lin_lut;
    float *tonemap_lut;
    uint16_t *delin_lut;
    int in_yuv_off, out_yuv_off;
    int16_t (*yuv2rgb_coeffs)[3][3][8];
    int16_t (*rgb2yuv_coeffs)[3][3][8];
    double  (*rgb2rgb_coeffs)[3][3];
    const struct LumaCoefficients *coeffs, *ocoeffs;
    double desat;
} TonemapIntParams;

typedef struct TonemapContext {
    const AVClass *class;

    enum TonemapAlgorithm tonemap;
    double param;
    double desat;
    double peak;

    const struct LumaCoefficients *coeffs, *ocoeffs;

    void (*tonemap_frame_p010_nv12)(uint8_t *dsty, uint8_t *dstuv, const uint16_t *src, const uint16_t *srcuv, const int *dstlinesize, const int *srclinesize, int width, int height, const struct TonemapIntParams *params);

    double lut_peak;
    float *lin_lut;
    float *tonemap_lut;
    uint16_t *delin_lut;
    int in_yuv_off, out_yuv_off;

    DECLARE_ALIGNED(16, int16_t, yuv2rgb_coeffs)[3][3][8];
    DECLARE_ALIGNED(16, int16_t, rgb2yuv_coeffs)[3][3][8];
    DECLARE_ALIGNED(16, double,  rgb2rgb_coeffs)[3][3];
} TonemapContext;

void ff_tonemap_frame_p010_nv12_c(uint8_t *dsty, uint8_t *dstuv, const uint16_t *src, const uint16_t *srcuv, const int *dstlinesize, const int *srclinesize, int width, int height, const struct TonemapIntParams *params);

#endif /* AVFILTER_GRADFUN_H */
