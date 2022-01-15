/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with FFmpeg; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <string.h>
#include "checkasm.h"
#include "libavfilter/yadif.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/mem_internal.h"

#define WIDTH 64
#define HEIGHT 8
#define SIZE WIDTH * HEIGHT + 32

#define randomize_buffers(buf, size)      \
    do {                                  \
        int j;                            \
        uint8_t *tmp_buf = (uint8_t*)buf; \
        for (j = 0; j < size; j++)        \
            tmp_buf[j] = rnd() & 0xFF;    \
    } while (0)

#define randomize_buffers_10(buf, size)    \
    do {                                   \
        int j;                             \
        uint16_t *tmp_buf = (uint16_t*)buf;\
        for (j = 0; j < size / 2; j++)     \
            tmp_buf[j] = rnd() & 0x03FF;   \
    } while (0)

static void check_yadif(enum AVPixelFormat fmt, int imode, const char * report_name)
{
    LOCAL_ALIGNED_32(uint8_t, cur_buf,     [SIZE]);
    LOCAL_ALIGNED_32(uint8_t, prev_buf,    [SIZE]);
    LOCAL_ALIGNED_32(uint8_t, next_buf,    [SIZE]);
    LOCAL_ALIGNED_32(uint8_t, dst_ref_buf, [SIZE]);
    LOCAL_ALIGNED_32(uint8_t, dst_new_buf, [SIZE]);
    int w = WIDTH, refs = WIDTH, h = HEIGHT;
    int df, pix_3, edge;
    int y;
    YADIFContext s;

    declare_func(void, void *dst1, void *prev1, void *cur1, void *next1,
                 int w, int prefs, int mrefs, int parity, int mode);

    memset(cur_buf,     0,  SIZE);
    memset(prev_buf,    0,  SIZE);
    memset(next_buf,    0,  SIZE);
    memset(dst_ref_buf, 0,  SIZE);
    memset(dst_new_buf, 0,  SIZE);

    s.csp = av_pix_fmt_desc_get(fmt);
    df = (s.csp->comp[0].depth + 7) / 8;
    w /= df;
    pix_3 = 3 * df;
    edge = 3 + 8 / df - 1;

    if (s.csp->comp[0].depth == 10) {
        randomize_buffers_10(cur_buf,  WIDTH * HEIGHT);
        randomize_buffers_10(prev_buf, WIDTH * HEIGHT);
        randomize_buffers_10(next_buf, WIDTH * HEIGHT);
    } else {
        randomize_buffers(cur_buf,  WIDTH * HEIGHT);
        randomize_buffers(prev_buf, WIDTH * HEIGHT);
        randomize_buffers(next_buf, WIDTH * HEIGHT);
    }

    ff_yadif_init(&s);

    if (check_func(s.filter_line, "yadif_%s", report_name)) {
        for (int parity = 0; parity <= 1; parity++) {
            for (y = 0; y < HEIGHT; y++) {
                uint8_t *prev    = &prev_buf[y * refs];
                uint8_t *cur     = &cur_buf[y * refs];
                uint8_t *next    = &next_buf[y * refs];
                uint8_t *dst_ref = &dst_ref_buf[y * refs];
                uint8_t *dst_new = &dst_new_buf[y * refs];
                int     mode  = y == 1 || y + 2 == HEIGHT ? 2 : imode;
                call_ref(dst_ref + pix_3, prev + pix_3, cur + pix_3,
                         next + pix_3, w - edge,
                         y + 1 < h ? refs : -refs,
                         y ? -refs : refs,
                         parity, mode);
                call_new(dst_new + pix_3, prev + pix_3, cur + pix_3,
                         next + pix_3, w - edge,
                         y + 1 < h ? refs : -refs,
                         y ? -refs : refs,
                         parity, mode);
                s.filter_edges(dst_ref, prev, cur, next, w,
                               y + 1 < h ? refs : -refs,
                               y ? -refs : refs,
                               parity, mode);
                s.filter_edges(dst_new, prev, cur, next, w,
                               y + 1 < h ? refs : -refs,
                               y ? -refs : refs,
                               parity, mode);
            }
            if (memcmp(dst_new_buf, dst_ref_buf, WIDTH * HEIGHT))
                fail();
        }
        bench_new(dst_new_buf + pix_3, prev_buf + pix_3,
                  cur_buf + pix_3, next_buf + pix_3,
                  w - edge, WIDTH, WIDTH, 0, imode);
    }
}
void checkasm_check_vf_yadif(void)
{
    check_yadif(AV_PIX_FMT_YUV420P, 0, "8");
    report("yadif_8");

    check_yadif(AV_PIX_FMT_YUV420P, 2, "8_nospatial");
    report("yadif_8_nospatial");

    check_yadif(AV_PIX_FMT_YUV420P10, 0, "10");
    report("yadif_10");

    check_yadif(AV_PIX_FMT_YUV420P10, 2, "10_nospatial");
    report("yadif_10_nospatial");

    check_yadif(AV_PIX_FMT_YUV420P16, 0, "16");
    report("yadif_16");

    check_yadif(AV_PIX_FMT_YUV420P16, 2, "16_nospatial");
    report("yadif_16_nospatial");
}
