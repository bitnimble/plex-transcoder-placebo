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
#include "libavfilter/bwdif.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/mem_internal.h"

#define WIDTH 64
#define HEIGHT 16
#define SIZE WIDTH * HEIGHT + 32

#define randomize_buffers(buf, size)      \
    do {                                  \
        int j;                            \
        uint8_t *tmp_buf = (uint8_t*)buf; \
        for (j = 0; j < size; j++)        \
            tmp_buf[j] = rnd() & 0xFF;    \
    } while (0)

#define randomize_buffers_12(buf, size)    \
    do {                                   \
        int j;                             \
        uint16_t *tmp_buf = (uint16_t*)buf;\
        for (j = 0; j < size / 2; j++)     \
            tmp_buf[j] = rnd() & 0x0FFF;   \
    } while (0)

static void check_bwdif(enum AVPixelFormat fmt, int imode, const char * report_name)
{
    LOCAL_ALIGNED_32(uint8_t, cur_buf,     [SIZE]);
    LOCAL_ALIGNED_32(uint8_t, prev_buf,    [SIZE]);
    LOCAL_ALIGNED_32(uint8_t, next_buf,    [SIZE]);
    LOCAL_ALIGNED_32(uint8_t, dst_ref_buf, [SIZE]);
    LOCAL_ALIGNED_32(uint8_t, dst_new_buf, [SIZE]);
    int clip_max;
    int w = WIDTH, refs;
    int df;
    int y;
    BWDIFContext s;
    YADIFContext *yadif = &s.yadif;

    declare_func(void, void *dst, void *prev, void *cur, void *next,
                 int w, int prefs, int mrefs, int prefs2, int mrefs2,
                 int prefs3, int mrefs3, int prefs4, int mrefs4,
                 int parity, int clip_max);

    memset(cur_buf,     0,  SIZE);
    memset(prev_buf,    0,  SIZE);
    memset(next_buf,    0,  SIZE);
    memset(dst_ref_buf, 0,  SIZE);
    memset(dst_new_buf, 0,  SIZE);

    yadif->csp = av_pix_fmt_desc_get(fmt);
    df = (yadif->csp->comp[0].depth + 7) / 8;
    w /= df;
    refs = w;
    clip_max = (1 << (yadif->csp->comp[0].depth)) - 1;

    if (yadif->csp->comp[0].depth == 12) {
        randomize_buffers_12(cur_buf,  WIDTH * HEIGHT);
        randomize_buffers_12(prev_buf, WIDTH * HEIGHT);
        randomize_buffers_12(next_buf, WIDTH * HEIGHT);
    } else {
        randomize_buffers(cur_buf,  WIDTH * HEIGHT);
        randomize_buffers(prev_buf, WIDTH * HEIGHT);
        randomize_buffers(next_buf, WIDTH * HEIGHT);
    }

    ff_bwdif_init(&s);

    if (check_func(s.filter_line, "bwdif_%s", report_name)) {
        for (int parity = 0; parity <= 1; parity++) {
            for (y = 4; y < HEIGHT - 4; y++) {
                uint8_t *prev    = &prev_buf[y * WIDTH];
                uint8_t *cur     = &cur_buf[y * WIDTH];
                uint8_t *next    = &next_buf[y * WIDTH];
                uint8_t *dst_ref = &dst_ref_buf[y * WIDTH];
                uint8_t *dst_new = &dst_new_buf[y * WIDTH];
                call_ref(dst_ref, prev, cur, next, w,
                         refs, -refs, refs << 1, -(refs << 1),
                         3 * refs, -3 * refs, refs << 2, -(refs << 2),
                         parity, clip_max);
                call_new(dst_new, prev, cur, next, w,
                         refs, -refs, refs << 1, -(refs << 1),
                         3 * refs, -3 * refs, refs << 2, -(refs << 2),
                         parity, clip_max);
            }
            if (memcmp(dst_new_buf, dst_ref_buf, WIDTH * HEIGHT))
                fail();
        }
        bench_new(dst_new_buf + (WIDTH * 4), prev_buf + (WIDTH * 4),
                  cur_buf + (WIDTH * 4), next_buf + (WIDTH * 4), w,
                  refs, -refs, refs << 1, -(refs << 1),
                  3 * refs, -3 * refs, refs << 2, -(refs << 2),
                  0, clip_max);
    }
}
void checkasm_check_vf_bwdif(void)
{
    check_bwdif(AV_PIX_FMT_YUV420P, 0, "8");
    report("bwdif_8");

    check_bwdif(AV_PIX_FMT_YUV420P12, 0, "12");
    report("bwdif_12");
}
