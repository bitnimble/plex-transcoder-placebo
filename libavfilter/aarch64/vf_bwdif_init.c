/*
 * Copyright (C) 2019 rcombs <rcombs@rcombs.me>
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

#include "libavutil/aarch64/cpu.h"
#include "libavfilter/bwdif.h"

void ff_bwdif_filter_line_neon(void *dst, void *prev, void *cur, void *next,
                               int w, int prefs, int mrefs, int prefs2,
                               int mrefs2, int prefs3, int mrefs3, int prefs4,
                               int mrefs4, int parity, int clip_max);

av_cold void ff_bwdif_init_aarch64(BWDIFContext *bwdif)
{
    YADIFContext *yadif = &bwdif->yadif;
    int cpu_flags = av_get_cpu_flags();
    int bit_depth = (!yadif->csp) ? 8
                                  : yadif->csp->comp[0].depth;

    if (bit_depth <= 8) {
        if (have_neon(cpu_flags))
            bwdif->filter_line = ff_bwdif_filter_line_neon;
    }
}
