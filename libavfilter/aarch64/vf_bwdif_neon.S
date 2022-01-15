/*
 * Copyright (C) 2006 Michael Niedermayer <michaelni@gmx.at>
 * Copyright (c) 2013 Daniel Kang <daniel.d.kang@gmail.com>
 * Copyright (C) 2016 Thomas Mundt <loudmax@yahoo.de>
 * Copyright (c) 2019 rcombs <rcombs@rcombs.me>
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

/*
 * void ff_bwdif_filter_line_neon(void *dst, void *prev, void *cur, void *next,
 *                                int w, int prefs, int mrefs, int prefs2,
 *                                int mrefs2, int prefs3, int mrefs3, int prefs4,
 *                                int mrefs4, int parity, int clip_max);
 */

#define dst    x0
#define prev   x1
#define cur    x2
#define next   x3
#define w      x4
#define prefs  x5
#define mrefs  x6
#define prefs2 x7
#define mrefs2 x8
#define prefs3 x9
#define mrefs3 x10
#define prefs4 x11
#define mrefs4 x12
#define parity w13
#define clip_max w14

#define coefhf0  v25
#define coefhf   v26
#define coefhf1  v27
#define coefsp   v28
#define coefsp1  v29
#define splfdif  v30
#define splfdif1 v31

.macro LOAD8 dst, src, offset:vararg
.ifnb \offset
    add      x13, \src, \offset
    ld1      {\dst\().8B}, [x13]
.else
    ld1      {\dst\().8B}, [\src]
.endif
    zip1     \dst\().16B, \dst\().16B, v7.16B
.endm

.macro DISP8 src
    sqxtun       \src\().8B, \src\().8H
    st1          {\src\().8B}, [dst]
.endm

.macro FILTER parity, a, b, bits, bytes
1:
    LOAD\bits    v0, cur, prefs, LSL #(\bytes - 1)
    LOAD\bits    v1, cur, mrefs, LSL #(\bytes - 1)
    LOAD\bits    v2, \a
    LOAD\bits    v4, \b

    add          v16.8H, v4.8H, v2.8H
    uabd         v17.8H, v2.8H, v4.8H

    LOAD\bits    v3, prev, prefs, LSL #(\bytes - 1)
    LOAD\bits    v4, prev, mrefs, LSL #(\bytes - 1)

    uabd         v3.8H, v3.8H, v0.8H
    uabd         v4.8H, v4.8H, v1.8H
    add          v3.8H, v3.8H, v4.8H
    umax         v2.8H, v17.8H, v3.8H

    LOAD\bits    v3, next, prefs, LSL #(\bytes - 1)
    LOAD\bits    v4, next, mrefs, LSL #(\bytes - 1)

    uabd         v3.8H, v3.8H, v0.8H
    uabd         v4.8H, v4.8H, v1.8H
    add          v3.8H, v3.8H, v4.8H
    umax         v2.8H, v2.8H, v3.8H
    ushr         v2.8H, v2.8H, 1

    LOAD\bits    v3, \a, prefs, LSL \bytes
    LOAD\bits    v4, \b, prefs, LSL \bytes
    LOAD\bits    v5, \a, mrefs, LSL \bytes
    LOAD\bits    v6, \b, mrefs, LSL \bytes

    add          v3.8H, v3.8H, v4.8H
    add          v5.8H, v5.8H, v6.8H
    add          v18.8H, v3.8H, v5.8H

    ushr         v3.8H, v3.8H, 1
    ushr         v5.8H, v5.8H, 1
    sub          v6.8H, v3.8H, v0.8H
    sub          v5.8H, v5.8H, v1.8H
    smin         v3.8H, v6.8H, v5.8H
    smax         v5.8H, v5.8H, v6.8H
    sshr         v4.8H, v16.8H, 1
    sub          v6.8H, v4.8H, v0.8H
    sub          v4.8H, v4.8H, v1.8H

    smax         v3.8H, v3.8H, v6.8H
    smin         v5.8H, v5.8H, v6.8H
    smax         v3.8H, v3.8H, v4.8H
    smin         v5.8H, v5.8H, v4.8H

    sub          v6.8H, v7.8H, v3.8H
    smax         v6.8H, v6.8H, v5.8H
    cmgt         v3.8H, v2.8H, v7.8H
    and          v6.16B, v6.16B, v3.16B
    smax         v19.8H, v2.8H, v6.8H

    LOAD\bits    v2, \a, prefs, LSL #(\bytes + 1)
    LOAD\bits    v3, \b, prefs, LSL #(\bytes + 1)
    LOAD\bits    v4, \a, mrefs, LSL #(\bytes + 1)
    LOAD\bits    v5, \b, mrefs, LSL #(\bytes + 1)

    add          v2.8H, v2.8H, v3.8H
    add          v4.8H, v4.8H, v5.8H
    add          v3.8H, v2.8H, v4.8H

    smull        v2.4S, v3.4H, coefhf0.4H
    smull2       v3.4S, v3.8H, coefhf0.8H
    smlal        v2.4S, v16.4H, coefhf.4H
    smlal2       v3.4S, v16.8H, coefhf.8H

    smull        v4.4S, v18.4H, coefhf1.4H
    smull2       v5.4S, v18.8H, coefhf1.8H

    add          v2.4S, v2.4S, v4.4S
    add          v3.4S, v3.4S, v5.4S
    sshr         v2.4S, v2.4S, 2
    sshr         v3.4S, v3.4S, 2

    add          v4.8H, v0.8H, v1.8H

    LOAD\bits    v5, cur, prefs3, LSL #(\bytes - 1)
    LOAD\bits    v6, cur, mrefs3, LSL #(\bytes - 1)

    add          v6.8H, v6.8H, v5.8H
    uabd         v1.8H, v1.8H, v0.8H
    cmgt         v1.8H, v1.8H, v17.8H

    zip1         v21.8H, v1.8H, v1.8H
    zip2         v0.8H, v1.8H, v1.8H

    and          v2.16B, v2.16B, v21.16B
    and          v3.16B, v3.16B, v0.16B

    and          v5.16B, v1.16B, splfdif1.16B
    and          v20.16B, v1.16B, splfdif.16B
    add          v5.8H, v5.8H, coefsp1.8H
    add          v20.8H, v20.8H, coefsp.8H

    smull        v0.4S, v4.4H, v20.4H
    smull2       v4.4S, v4.8H, v20.8H
    smlal        v0.4S, v6.4H, v5.4H
    smlal2       v4.4S, v6.8H, v5.8H

    add          v2.4S, v2.4S, v0.4S
    add          v3.4S, v3.4S, v4.4S
    sqshrn       v2.4H, v2.4S, 13
    sqshrn2      v2.8H, v3.4S, 13

    sshr         v4.8H, v16.8H, 1
    add          v3.8H, v4.8H, v19.8H
    sub          v4.8H, v4.8H, v19.8H
    smax         v2.8H, v2.8H, v4.8H
    smin         v2.8H, v2.8H, v3.8H
    DISP\bits    v2

    add          dst,  dst,  8 * \bytes
    add          prev, prev, 8 * \bytes
    add          cur,  cur,  8 * \bytes
    add          next, next, 8 * \bytes
    subs         w,    w,    8
    b.gt 1b
.endm

.macro PROC bits, bytes
    uxtw            x4, w4               // Zero-extend w to 64-bit
    sxtw            x5, w5               // Sign-extend prefs and mrefs to 64-bit
    sxtw            x6, w6
    sxtw            x7, w7
    ldr             w8, [sp]             // Load and zero-extend stack-passed int args
    sxtw            x8, w8
    ldr             w9, [sp, 8]
    sxtw            x9, w9
    ldr             w10, [sp, 16]
    sxtw            x10, w10
    ldr             w11, [sp, 24]
    sxtw            x11, w11
    ldr             w12, [sp, 32]
    sxtw            x12, w12
    ldr             parity, [sp, 40]

    movi            v7.4S, #0            // Set up some static-value registers
    mov             w14, 1016
    dup             coefhf0.8H, w14
    mov             w14, 5570
    dup             coefhf.8H, w14
    mov             w14, -3801
    dup             coefhf1.8H, w14
    mov             w14, 5077
    dup             coefsp.8H, w14
    mov             w14, -981
    dup             coefsp1.8H, w14
    mov             w14, -768
    dup             splfdif.8H, w14
    mov             w14, 768
    dup             splfdif1.8H, w14

    cbz             parity, 0f                   // If parity was 0, take that branch
    FILTER 1, prev, cur, \bits, \bytes
    ret
0:  FILTER 0, cur, next, \bits, \bytes
    ret
.endm

/*
 * void ff_bwdif_filter_line_neon(void *dst, void *prev, void *cur, void *next,
 *                                int w, int prefs, int mrefs, int prefs2,
 *                                int mrefs2, int prefs3, int mrefs3, int prefs4,
 *                                int mrefs4, int parity, int clip_max);
 */

function ff_bwdif_filter_line_neon, export=1
    PROC 8, 1
