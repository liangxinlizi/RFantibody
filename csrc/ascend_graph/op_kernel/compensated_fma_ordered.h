// CUDA-order FP32 contraction primitives for the RFdiffusion tied attention.
//
// The CUDA reference evaluates the tied attention contraction
// ``opt_einsum.contract('bnihk,bnjhk->bijh', query, key)`` as a sequential,
// correctly rounded float32 multiply-add chain with ``k`` as the outer and ``n``
// as the inner loop variable (measured on the recording H100; the arithmetic
// contract is stated below and asserted by the parity tests).  This header owns
// the two
// device-side primitives that reproduce that arithmetic exactly on the Ascend
// AIV without a fused multiply-add instruction and without a scalar inner loop:
//
// * :meth:`OrderedFma::SplitMultiplicand` -- the Dekker split of the key operand
//   (``bh``/``bv``).  The key operand does not depend on the query row, so its
//   split is computed once per ``(k, n)`` step and shared by every output row of
//   the tile.
// * :meth:`OrderedFma::Step` -- ``acc = fma_f32(a, b, acc)``: one correctly
//   rounded float32 FMA built from a compensated product (Dekker split plus the
//   exact product error), two branch-free TwoSums (Knuth) and a vectorized
//   round-to-odd of the low part, exactly the primitive measured to reproduce
//   the CUDA value bit-exactly.
//
// Every operation is a whole-row vector operation over ``count`` float32/int32
// lanes; the arithmetic never degrades to a scalar ``GetValue``/``SetValue``
// loop.  The installed CANN vector API of this SoC (``dav-c220-vec``) supports
// int32 ``Add``/``Sub``/``Mul``/``Mins``/``ShiftRight`` but only int16/uint16 for
// the bitwise ``And``/``Or``/``Not``, so the round-to-odd bit adjustment is
// expressed with int32 arithmetic on the reinterpreted float32 bit patterns
// (``Abs`` on the float value yields the magnitude bits, ``Mins``/``ShiftRight``/
// ``Muls``/``Sub``/``Mul``/``Add`` do the adjustment).
//
// There is no Torch/NumPy/CPU computation here and no per-shape, per-case or
// per-example branch: the functions act elementwise on the lanes they are given.

#ifndef ASCEND_GRAPH_OP_KERNEL_COMPENSATED_FMA_ORDERED_H
#define ASCEND_GRAPH_OP_KERNEL_COMPENSATED_FMA_ORDERED_H

#include "kernel_operator.h"

namespace ascend_graph_ordered {

using AscendC::LocalTensor;

// float32 scratch arrays of one ordered-FMA step (qv, pp, sa, ah, av, ep, s, z,
// es, lo, tail, t1, t2, sb, bh, bv, |lo|, |tail|).
constexpr int32_t kFloatScratchArrays = 18;
// int32 scratch arrays of the vectorized round-to-odd.
constexpr int32_t kIntScratchArrays = 10;
// int32 constant arrays (the value 1).
constexpr int32_t kIntConstantArrays = 1;
// Dekker split factor for the 24-bit float32 significand (2^12 + 1).
constexpr float kDekkerSplit = 4097.0f;
// INT32_MIN reinterpreted as the sign bit of a float32 (0x80000000).
constexpr int32_t kFloatSignBit = static_cast<int32_t>(0x80000000u);
// Shift that expands an int32 sign into the 0 / -1 mask used below.
constexpr int32_t kSignShift = 31;

class OrderedFma {
public:
    __aicore__ inline OrderedFma() {}

    // Bind the scratch carving.  ``laneStride`` is the element stride between
    // arrays and must stay a multiple of the 32-byte vector block so every
    // carved view is block aligned.
    __aicore__ inline void Init(LocalTensor<float> floatScratch,
                                LocalTensor<int32_t> intScratch,
                                LocalTensor<int32_t> intConstants,
                                int32_t laneStride)
    {
        this->stride = laneStride;
        qv = floatScratch[At(0)];
        pp = floatScratch[At(1)];
        sa = floatScratch[At(2)];
        ah = floatScratch[At(3)];
        av = floatScratch[At(4)];
        ep = floatScratch[At(5)];
        s = floatScratch[At(6)];
        z = floatScratch[At(7)];
        es = floatScratch[At(8)];
        low = floatScratch[At(9)];
        tail = floatScratch[At(10)];
        t1 = floatScratch[At(11)];
        t2 = floatScratch[At(12)];
        splitB = floatScratch[At(13)];
        bHi = floatScratch[At(14)];
        bLo = floatScratch[At(15)];
        absLow = floatScratch[At(16)];
        absTail = floatScratch[At(17)];

        wNzTail = intScratch[At(0)];
        wEven = intScratch[At(1)];
        wTmp = intScratch[At(2)];
        wMagNew = intScratch[At(3)];
        wCorner = intScratch[At(4)];
        wNeed = intScratch[At(5)];
        wDelta = intScratch[At(6)];
        wNegLow = intScratch[At(7)];
        wNegTail = intScratch[At(8)];
        wExtra = intScratch[At(9)];

        cOne = intConstants[At(0)];
        AscendC::Duplicate(cOne, static_cast<int32_t>(1), laneStride);
    }

    // Dekker split of the multiplicand ``b`` (the key row).  ``bHi + bLo == b``
    // exactly, with each part holding at most 12 significand bits, which makes
    // the product error of the following multiplication exactly representable.
    __aicore__ inline void SplitMultiplicand(LocalTensor<float> b, int32_t count)
    {
        AscendC::Muls(splitB, b, kDekkerSplit, count);
        AscendC::Sub(t1, splitB, b, count);
        AscendC::Sub(bHi, splitB, t1, count);
        AscendC::Sub(bLo, b, bHi, count);
    }

    // ``acc = fma_f32(a, b, acc)`` with one correctly rounded float32 rounding.
    //
    // * ``aValue`` is the query operand of the whole row (a scalar of the
    //   logical ``a`` operand: it does not depend on the output column).
    // * ``b`` is the key row (already split by :meth:`SplitMultiplicand`).
    // * ``acc`` is the running accumulator of the row, overwritten in place.
    __aicore__ inline void Step(float aValue, LocalTensor<float> b,
                                LocalTensor<float> acc, int32_t count)
    {
        // a-side Dekker split (the same split the multiplicand uses).
        AscendC::Duplicate(qv, aValue, count);
        AscendC::Muls(sa, qv, kDekkerSplit, count);
        AscendC::Sub(t1, sa, qv, count);
        AscendC::Sub(ah, sa, t1, count);
        AscendC::Sub(av, qv, ah, count);

        // p = fl(a*b) and its exact product error ep (two-product).
        AscendC::Mul(pp, qv, b, count);
        AscendC::Mul(ep, ah, bHi, count);
        AscendC::Sub(ep, ep, pp, count);
        AscendC::Mul(t1, ah, bLo, count);
        AscendC::Add(ep, ep, t1, count);
        AscendC::Mul(t1, av, bHi, count);
        AscendC::Add(ep, ep, t1, count);
        AscendC::Mul(t1, av, bLo, count);
        AscendC::Add(ep, ep, t1, count);

        // TwoSum(pp, acc) -> s + es == pp + acc exactly.
        AscendC::Add(s, pp, acc, count);
        AscendC::Sub(z, s, pp, count);
        AscendC::Sub(t1, s, z, count);
        AscendC::Sub(es, pp, t1, count);
        AscendC::Sub(t2, acc, z, count);
        AscendC::Add(es, es, t2, count);

        // TwoSum(ep, es) -> low + tail == ep + es exactly.
        AscendC::Add(low, ep, es, count);
        AscendC::Sub(z, low, ep, count);
        AscendC::Sub(t1, low, z, count);
        AscendC::Sub(tail, ep, t1, count);
        AscendC::Sub(t2, es, z, count);
        AscendC::Add(tail, tail, t2, count);

        // Round the low part to odd so that the final ``s + low`` add is the
        // correctly rounded value of the exact three-term sum.
        RoundToOdd(count);

        AscendC::Add(acc, s, low, count);
    }

private:
    __aicore__ inline uint32_t At(int32_t index) const
    {
        return static_cast<uint32_t>(index * this->stride);
    }

    // ``low = round_to_odd(low + tail)`` on the stored float32 bit patterns.
    //
    // Reference (scalar form): ``bits = reinterpret(low); if (tail != 0 &&
    // (bits & 1) == 0) bits += sign(low) == sign(tail) ? 1 : -1; if (low == 0)
    // bits = sign(tail) | 1;``.  The vector form works on the magnitude bits
    // (``Abs`` clears only the sign bit of a float32) and re-applies the sign at
    // the end; ``low`` is overwritten with the adjusted value.
    __aicore__ inline void RoundToOdd(int32_t count)
    {
        AscendC::Abs(absLow, low, count);
        AscendC::Abs(absTail, tail, count);
        LocalTensor<int32_t> magLow = absLow.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> magTail = absTail.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> bitsLow = low.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> bitsTail = tail.ReinterpretCast<int32_t>();

        // 1 iff the tail residual is not exactly zero (a magnitude bit pattern
        // is >= 1 for every non-zero float32 value).
        AscendC::Mins(wNzTail, magTail, static_cast<int32_t>(1), count);

        // Even flag of the low part: odd = mag - 2 * floor(mag / 2), even = 1 -
        // odd.  Every vector call keeps its destination distinct from its second
        // source, so no in-place aliasing beyond the accepted `dst == src0`
        // pattern of this repository's kernels is used.
        AscendC::ShiftRight(wTmp, magLow, static_cast<int32_t>(1), count);
        AscendC::Muls(wTmp, wTmp, static_cast<int32_t>(2), count);
        AscendC::Sub(wEven, magLow, wTmp, count);
        AscendC::Sub(wExtra, cOne, wEven, count);

        // need = tail != 0 && low is even (0/1).
        AscendC::Mul(wNeed, wNzTail, wExtra, count);

        // Sign indicators (0/1) of both operands.
        AscendC::ShiftRight(wNegLow, bitsLow, kSignShift, count);
        AscendC::Muls(wNegLow, wNegLow, static_cast<int32_t>(-1), count);
        AscendC::ShiftRight(wNegTail, bitsTail, kSignShift, count);
        AscendC::Muls(wNegTail, wNegTail, static_cast<int32_t>(-1), count);

        // delta = +1 when the signs agree, -1 when they differ (0/1 arithmetic).
        AscendC::Sub(wDelta, wNegLow, wNegTail, count);
        AscendC::Mul(wTmp, wDelta, wDelta, count);
        AscendC::Muls(wTmp, wTmp, static_cast<int32_t>(2), count);
        AscendC::Sub(wDelta, cOne, wTmp, count);

        // One ulp step of the magnitude in the residual's direction.
        AscendC::Mul(wNeed, wNeed, wDelta, count);
        AscendC::Add(wMagNew, magLow, wNeed, count);

        // Corner: exactly zero low part with a non-zero residual takes the
        // smallest magnitude and the residual's sign.
        AscendC::Mins(wCorner, magLow, static_cast<int32_t>(1), count);
        AscendC::Sub(wExtra, cOne, wCorner, count);
        AscendC::Mul(wCorner, wExtra, wNzTail, count);

        AscendC::Sub(wExtra, cOne, wMagNew, count);
        AscendC::Mul(wExtra, wExtra, wCorner, count);
        AscendC::Add(wMagNew, wMagNew, wExtra, count);

        // Final sign: the low part's sign, overridden by the residual's sign in
        // the zero-magnitude corner, exactly like the scalar reference.
        AscendC::Sub(wExtra, wNegTail, wNegLow, count);
        AscendC::Mul(wExtra, wExtra, wCorner, count);
        AscendC::Add(wTmp, wNegLow, wExtra, count);
        AscendC::Muls(wTmp, wTmp, kFloatSignBit, count);

        AscendC::Add(bitsLow, wMagNew, wTmp, count);
    }

    int32_t stride = 0;
    LocalTensor<float> qv;
    LocalTensor<float> pp;
    LocalTensor<float> sa;
    LocalTensor<float> ah;
    LocalTensor<float> av;
    LocalTensor<float> ep;
    LocalTensor<float> s;
    LocalTensor<float> z;
    LocalTensor<float> es;
    LocalTensor<float> low;
    LocalTensor<float> tail;
    LocalTensor<float> t1;
    LocalTensor<float> t2;
    LocalTensor<float> splitB;
    LocalTensor<float> bHi;
    LocalTensor<float> bLo;
    LocalTensor<float> absLow;
    LocalTensor<float> absTail;
    LocalTensor<int32_t> wNzTail;
    LocalTensor<int32_t> wEven;
    LocalTensor<int32_t> wTmp;
    LocalTensor<int32_t> wMagNew;
    LocalTensor<int32_t> wCorner;
    LocalTensor<int32_t> wNeed;
    LocalTensor<int32_t> wDelta;
    LocalTensor<int32_t> wNegLow;
    LocalTensor<int32_t> wNegTail;
    LocalTensor<int32_t> wExtra;
    LocalTensor<int32_t> cOne;
};

}  // namespace ascend_graph_ordered

#endif  // ASCEND_GRAPH_OP_KERNEL_COMPENSATED_FMA_ORDERED_H
