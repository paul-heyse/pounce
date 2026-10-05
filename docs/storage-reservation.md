# Serial AMD storage reservation

This source-owned reservation applies to the regular FERAL serial AMD route with
identity, infinity-norm, automatic or symmetric MC64 scaling. It preserves the
normal sparse/dense dispatch and numerical quality policy. Parallel factors,
and external orderings are opaque.
A dimension ceiling is optional; overflow is a terminal resource refusal before
allocation. The coefficients intentionally overreserve and are not measurements.

Let n be the greater of the new and retained factor dimensions, z the greater of
new and retained input-triplet counts, and r the greater of the requested and
retained RHS widths (at least n at pattern admission). Every original column can
own one front, and every front contains at most n original columns: delayed
pivots move columns without creating additional columns. Thus each dense front,
contribution and frontal factor has at most n² scalars and a family has at most
n³. Serial factorization has no concurrently executing worker fronts.

`Solver::conservative_storage_bytes` reserves two times the scalar and index
counts below, covering geometric Vec growth and old/new capacities. Headers and
minimum allocations are separately reserved per original column.

| Owners inspected | Scalar allowance | Index allowance |
| --- | ---: | ---: |
| Retained factor, current factor, retry/escalation candidate, frontal workspace, contributions, factor result replacement (`numeric/solver.rs`, `numeric/factorize.rs`) | 48n³ | 16n³ |
| Dense fast paths, factor packing (MR/NR padding), workspace permutation caches, canonical/scaled CSC, symbolic replacements, MC64 costs and scaling | 96n² | 192n² + 256n |
| Packed RHS, original RHS copies, solve result, sparse triangular workspace, residual/correction and refinement replacements (`numeric/solve.rs`, `numeric/refine.rs`) | 64nr | included above |

The sparse-factor cubic allowance assigns eight n³ scalar families to each of
six simultaneously live owner groups; the index allowance assigns two n³ index
families to eight groups. A node owns its L and contribution arrays plus linear
D/permutation/row arrays. Workspace owns one dense front, one factor scratch and
one contribution scratch. Retry and ordering escalation execute serially and
retain only the prior/current results, not all attempted factors. These groups
include the retained result during replacement and the stale warm workspace.
Dense packing sizes are ceil(rows/MR)*MR*panel and
ceil(cols/NR)*NR*panel: the square allowance plus linear/minimum header allowance
covers their tile padding. No factor route is changed to obtain this bound.

Symmetric MC64 builds a full cost graph (at most n² entries), cost/row arrays and
column offsets; matching uses indexed heaps with at most n entries, and its
search/permutation/dual vectors have length n. Symbolic MC64 caches and new
matching results can coexist. The AMD adapter builds full adjacency; resolved
feral-amd 0.2.1 uses feral-ordering-core 0.2.2 quotient-graph workspace with
`iwlen = nzaat + nzaat/5 + n`, an arena compacted in place and ten length-n
integer vectors. Final postorder/permutation scratch is linear. The index square
allowance includes these graphs, the symbolic permuted pattern and all copied
CSC/permutation/value maps, using usize width even for i32 storage.

`pounce-feral::bounded_factor_storage` adds wrapper triplet and slot arrays,
CSC construction pairs/offsets and old/new matrix/result buffers in terms of z.
A backsolve updates the same owner reservation before resizing RHS scratch and
includes retained wider RHS capacity. `bounded_schur_storage` adds two complete
factor reservations, all partition and triplet maps, retained/replacement FF/SS
matrices, coupling AFS, W, ASTW, dense separator, Schur CSC, packed BF/BS/RS/RHSF/
XS/XF and iterative refinement scratch. Both factors may be live together.
The assembled KKT wrapper separately admits retained and replacement triplets,
packing, flattening and scaling vectors. Algorithm bundle headers are admitted
before construction. A fallback admits the complete monolithic owner separately
while the Schur owner is still live. The observer sums distinct owner identities;
its refusal is sticky and cannot be interpreted as numerical fallback.

The observer consumer must retain reservations through factor replacement and
teardown. On a compatible new task it can transfer allocation leases separately
from a completed task's work ledger; old and new leases overlap until the
replacement owner has been admitted. The Woodbury owner admits its actual rank and retained rank before allocating
Vtilde/Utilde, dense J1/J2, rank cross-products and batched RHS/action copies.
Its two independent monolithic factors are separate source instances. The
restoration projection owner admits actual compound dimension, rank and triplet
counts before low-rank extraction or dense materialization, original Jacobian
copies, reduction vectors and action scratch. Their reserve_wrapper allowance
contains 128n² + 128nr + 64r² + 128z + 256n + 1024 machine words, doubled
for capacity/replacement overlap; these dominate the eight per-block
multivectors, six rank-product matrices, projected dense/triplet matrices and
all simultaneously live vector/correction/result copies. Inputs owned by the
Hessian updater itself are application storage and remain covered by its foreign
reservation, rather than being labeled a complete application estimate.
Ordinary unbounded execution may use opaque routes,
which cannot satisfy a complete linear-storage demand.
