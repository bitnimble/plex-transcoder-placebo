/*
 * Copyright (c) 2018 Clément Bœsch <u pkh me>
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

#include "libavutil/aarch64/asm.S"

#define dst    x0
#define prev   x1
#define cur    x2
#define next   x3
#define w      x4
#define prefs  x5
#define mrefs  x6
#define parity w7
#define mode   w7 // will be loaded here after parity check
#define prefs2 x10
#define mrefs2 x11

.macro CHECK a, b
    add       x9, cur, mrefs
.if \a
    add       x9, x9, \a
.endif
    ld1       {v2.16B}, [x9]
    add       x9, cur, prefs
.if \b
    add       x9, x9, \b
.endif
    ld1       {v3.16B}, [x9]

    eor       v4.16B, v2.16B, v3.16B
    urhadd    v5.16B, v2.16B, v3.16B
    and       v4.16B, v4.16B, v30.16B
    uqsub     v5.16B, v5.16B, v4.16B
    ext       v5.16B, v5.16B, v7.16B, 1
    zip1      v5.16B, v5.16B, v7.16B

    uqsub     v4.16B, v2.16B, v3.16B
    uqsub     v3.16B, v3.16B, v2.16B
    umax      v4.16B, v4.16B, v3.16B

    ext       v3.16B, v4.16B, v7.16B, 1
    ext       v2.16B, v4.16B, v7.16B, 2

    zip1      v4.16B, v4.16B, v7.16B
    zip1      v3.16B, v3.16B, v7.16B
    zip1      v2.16B, v2.16B, v7.16B

    add       v4.8H, v4.8H, v3.8H
    add       v4.8H, v4.8H, v2.8H
.endm

.macro CHECK1
    cmgt v6.8H, v19.8H, v4.8H
    smin v19.8H, v19.8H, v4.8H
    and  v5.16B, v5.16B, v6.16B
    bic  v3.16B, v18.16B, v6.16B
    orr  v18.16B, v3.16B, v5.16B
.endm

.macro CHECK2
    add     v6.8H, v6.8H, v31.8H
    shl     v6.8H, v6.8H, 14
    sqadd   v4.8H, v4.8H, v6.8H
    cmgt    v3.8H, v19.8H, v4.8H
    smin    v19.8H, v19.8H, v4.8H
    and     v5.16B, v5.16B, v3.16B
    bic     v3.16B, v18.16B, v3.16B
    orr     v18.16B, v3.16B, v5.16B
.endm

.macro LOAD dst, src, offset
.ifnb \offset
    add      x12, \src, \offset
    ld1      {\dst\().8B}, [x12]
.else
    ld1      {\dst\().8B}, [\src]
.endif
    zip1     \dst\().16B, \dst\().16B, v7.16B
.endm

.macro FILTER parity, prev2, next2
1:
    LOAD         v0, cur, mrefs
    LOAD         v1, cur, prefs
    LOAD         v2, \prev2
    LOAD         v4, \next2
    uhadd        v20.8H, v4.8H, v2.8H
//    mova   [rsp+ 0], m0
//    mova   [rsp+16], m20
//    mova   [rsp+32], m1
    uabd         v2.8H, v2.8H, v4.8H

    LOAD         v3, prev, mrefs
    LOAD         v4, prev, prefs
    uabd         v3.8H, v3.8H, v0.8H
    uaba         v3.8H, v4.8H, v1.8H
    umax         v2.8H, v2.8H, v3.8H

    LOAD         v3, next, mrefs
    LOAD         v4, next, prefs
    uabd         v3.8H, v3.8H, v0.8H
    uaba         v3.8H, v4.8H, v1.8H
    umax         v2.8H, v2.8H, v3.8H
    ushr         v17.8H, v2.8H, 1
    //mova   [rsp+48], m17

    add          v18.8H, v1.8H, v0.8H
    add          v19.8H, v0.8H, v0.8H
    uabd         v19.8H, v19.8H, v18.8H
    ushr         v18.8H, v18.8H, 1

    sub          x9, cur, 1
    add          x12, x9, mrefs
    ld1          {v4.16B}, [x12]
    add          x12, x9, prefs
    ld1          {v3.16B}, [x12]
    uqsub        v2.16B, v4.16B, v3.16B
    uqsub        v3.16B, v3.16B, v4.16B
    umax         v2.16B, v2.16B, v3.16B
    ext          v3.16B, v2.16B, v7.16B, 2
    zip1         v2.16B, v2.16B, v7.16B
    zip1         v3.16B, v3.16B, v7.16B
    add          v19.8H, v19.8H, v2.8H
    add          v19.8H, v19.8H, v3.8H
    sub          v19.8H, v19.8H, v31.8H

    CHECK -2, 0
    CHECK1
    CHECK -3, 1
    CHECK2
    CHECK 0, -2
    CHECK1
    CHECK 1, -3
    CHECK2

//    mova         m17, [rsp+48] // 17
    tbnz         mode, 1, 2f           // Skip the spatial section if not enabled
    LOAD         v2, \prev2, mrefs2
    LOAD         v4, \next2, mrefs2
    LOAD         v3, \prev2, prefs2
    LOAD         v5, \next2, prefs2

    uhadd        v2.8H, v2.8H, v4.8H
    uhadd        v3.8H, v3.8H, v5.8H

//    mova         m4, [rsp+ 0] // 0
//    mova         m20, [rsp+16] // 20
//    mova         m7, [rsp+32] // 1

    sub          v2.8H, v2.8H, v0.8H
    sub          v3.8H, v3.8H, v1.8H
    sub          v19.8H, v20.8H, v1.8H
    sub          v21.8H, v20.8H, v0.8H

    smin         v0.8H, v2.8H, v3.8H
    smax         v3.8H, v2.8H, v3.8H
    smax         v2.8H, v0.8H, v21.8H
    smin         v3.8H, v3.8H, v21.8H
    smax         v2.8H, v2.8H, v19.8H
    smin         v3.8H, v3.8H, v19.8H
    smax         v17.8H, v17.8H, v3.8H
    sub          v0.8H, v7.8H, v2.8H
    smax         v17.8H, v17.8H, v0.8H

2:
//    mova         m2, [rsp+16] // 17
    sub          v2.8H, v20.8H, v17.8H
    add          v3.8H, v20.8H, v17.8H
    smax         v18.8H, v18.8H, v2.8H
    smin         v18.8H, v18.8H, v3.8H
    uzp1         v18.16B, v18.16B, v18.16B

    st1          {v18.8B}, [dst]
    add          dst,  dst,  8
    add          prev, prev, 8
    add          cur,  cur,  8
    add          next, next, 8
    subs         w,    w,    8
    b.gt 1b
.endm

/*
 * void ff_yadif_filter_line_neon(void *dst, void *prev, void *cur,
 *                                void *next, int w, int prefs,
 *                                int mrefs, int parity, int mode);
 */
function ff_yadif_filter_line_neon, export=1
    uxtw            x4, w4               // Zero-extend w to 64-bit
    sxtw            x5, w5               // Sign-extend prefs and mrefs to 64-bit
    sxtw            x6, w6
    add             prefs2, prefs, prefs // Doubled versions of prefs and mrefs
    add             mrefs2, mrefs, mrefs
    movi            v7.4S, #0            // Set up some static-value registers
    movi            v30.16B, #1
    movi            v31.8H, #1
    cmp             parity, #0           // Check if parity is 0
    ldr             mode, [sp]           // We're finished with parity; use its register for the mode
    b.eq            0f                   // If parity was 0, take that branch
    FILTER 1, prev, cur
    ret
0:  FILTER 0, cur, next
    ret
endfunc
