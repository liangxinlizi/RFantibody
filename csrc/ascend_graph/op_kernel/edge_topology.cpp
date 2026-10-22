// Native AscendC edge-topology builder for the RFantibody graph operators.
//
// DGL 1.1.3 builds a CSR table (per-destination sorted incoming edge ids and
// in-degrees) before it can aggregate edge features.  RFantibody reaches that
// aggregation through `dgl.ops.copy_e_sum`, `dgl.ops.copy_e_mean`,
// `DGLGraph.in_degrees()` and the destination `edge_softmax`, so the migration
// needs a native, deterministic producer of exactly that table.  This kernel is
// it:
//
//   offsets [N+1] int64   offsets[s] .. offsets[s+1) is the CSR range of s
//   indices [E]   int32   original edge ids of destination s, ascending
//   degrees [N]   int64   in-degree of every destination
//
// A single AIV block performs a stable counting sort: first count the edges per
// destination, then turn the counts into a prefix sum (written both to
// `offsets` and to a scratch `cursor`), then place each edge id at its cursor
// position.  Because the placement loop visits edges in ascending original id
// and each destination's cursor only advances, the resulting per-destination
// order is the stable ascending edge id order.  No atomics and no cross-core
// reduction are involved, so the table is reproducible and the float32
// reduction order of `copy_e_sum`/`copy_e_mean` is deterministic.
//
// Every GM region this kernel touches is bound with `SetGlobalBuffer` before
// first use, and every output element is written by this kernel: `offsets[0..N]`,
// `degrees[0..N)` and `indices[0..E)`.  The host allocates the buffers and
// performs no initialization and no index computation, and there is no Torch,
// DGL, NumPy or CPU fallback on this path.  The single-block (scalar, serial)
// form is intentionally chosen for correctness first; it is not a performance
// claim, and the host rejects `numEdges > INT32_MAX` because `indices` is int32.
//
// A destination index outside `[0, numNodes)` is not a valid graph.  Instead of
// writing outside `offsets` (memory corruption) or silently dropping the edge,
// the builder closes the table with deterministic `-1` sentinels in the unused
// tail of `indices`; the reduction kernel then poisons the affected destination
// row with NaN, so an invalid index fails the parity gate loudly.

#include "kernel_operator.h"

namespace {

using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;

class KernelEdgeTopology {
public:
    __aicore__ inline KernelEdgeTopology() {}

    __aicore__ inline bool Init(GM_ADDR dst, GM_ADDR offsets, GM_ADDR indices,
                                GM_ADDR degrees, GM_ADDR cursor,
                                int64_t numEdges, int64_t numNodes)
    {
        // Deterministic single-block builder: only block 0 does the work.
        if (GetBlockIdx() != 0) {
            return false;
        }
        if (numNodes < 0 || numEdges < 0) {
            return false;
        }
        this->numEdges = numEdges;
        this->numNodes = numNodes;

        dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(dst),
                              static_cast<uint64_t>(numEdges));
        offsetsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(offsets),
                                  static_cast<uint64_t>(numNodes) + 1U);
        indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices),
                                  static_cast<uint64_t>(numEdges));
        degreesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(degrees),
                                  static_cast<uint64_t>(numNodes));
        cursorGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cursor),
                                 static_cast<uint64_t>(numNodes));
        return true;
    }

    __aicore__ inline void Process()
    {
        // 1. Zero the count slots, then count one incoming edge per dst entry.
        for (int64_t n = 0; n < this->numNodes; ++n) {
            offsetsGm.SetValue(static_cast<uint64_t>(n), static_cast<int64_t>(0));
        }
        bool invalidIndex = false;
        for (int64_t e = 0; e < this->numEdges; ++e) {
            const int64_t d = dstGm.GetValue(static_cast<uint64_t>(e));
            if (!ValidDestination(d)) {
                invalidIndex = true;
                continue;
            }
            offsetsGm.SetValue(static_cast<uint64_t>(d),
                               offsetsGm.GetValue(static_cast<uint64_t>(d)) + 1);
        }

        // 2. Exclusive prefix sum: counts become CSR offsets and cursors.
        int64_t running = 0;
        for (int64_t n = 0; n < this->numNodes; ++n) {
            const int64_t count =
                offsetsGm.GetValue(static_cast<uint64_t>(n));
            degreesGm.SetValue(static_cast<uint64_t>(n), count);
            offsetsGm.SetValue(static_cast<uint64_t>(n), running);
            cursorGm.SetValue(static_cast<uint64_t>(n), running);
            running += count;
        }

        // 3. Stable placement: ascending edge id within each destination.
        for (int64_t e = 0; e < this->numEdges; ++e) {
            const int64_t d = dstGm.GetValue(static_cast<uint64_t>(e));
            if (!ValidDestination(d)) {
                continue;
            }
            const int64_t position =
                cursorGm.GetValue(static_cast<uint64_t>(d));
            indicesGm.SetValue(static_cast<uint64_t>(position),
                               static_cast<int32_t>(e));
            cursorGm.SetValue(static_cast<uint64_t>(d), position + 1);
        }

        // 4. Close the table.  For a valid graph `running == numEdges`.  With an
        //    invalid destination the unused tail is filled with `-1` sentinels
        //    and attached to the last destination's range, so the reduction
        //    kernel deterministically poisons that row instead of reading
        //    uninitialized memory.
        if (invalidIndex && this->numNodes > 0) {
            for (int64_t p = running; p < this->numEdges; ++p) {
                indicesGm.SetValue(static_cast<uint64_t>(p),
                                   static_cast<int32_t>(-1));
            }
            offsetsGm.SetValue(static_cast<uint64_t>(this->numNodes),
                               this->numEdges);
        } else {
            offsetsGm.SetValue(static_cast<uint64_t>(this->numNodes), running);
        }
    }

private:
    __aicore__ inline bool ValidDestination(int64_t d) const
    {
        return d >= 0 && d < this->numNodes;
    }

    GlobalTensor<int64_t> dstGm;
    GlobalTensor<int64_t> offsetsGm;
    GlobalTensor<int32_t> indicesGm;
    GlobalTensor<int64_t> degreesGm;
    GlobalTensor<int64_t> cursorGm;
    int64_t numEdges = 0;
    int64_t numNodes = 0;
};

}  // namespace

// Entry point consumed by the host bridge; the generated host stub exports
// `aclrtlaunch_edge_topology`.
extern "C" __global__ __aicore__ void edge_topology(
    GM_ADDR dst, GM_ADDR offsets, GM_ADDR indices, GM_ADDR degrees,
    GM_ADDR cursor, int64_t numEdges, int64_t numNodes)
{
    KernelEdgeTopology op;
    if (!op.Init(dst, offsets, indices, degrees, cursor, numEdges, numNodes)) {
        return;
    }
    op.Process();
}
