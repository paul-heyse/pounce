# Observed execution source extension

This maintainer artifact extends POUNCE from upstream commit a31dab9c21a25ce0e5e8a93ab78f0250c9f0f0fd. `vendor/feral` contains the published MIT-licensed FERAL0.18.0 source. Its `.cargo_vcs_info.json` and original manifest retain upstream provenance (32c936b8d1f6a5d42ccf96b7dfb1c05f80e402da). The normalized manifest uses an internal path while retaining package version0.18.0.

## Admission and layout

The synchronous observer binds the actual postclassification TNLP layout before algorithm construction. Its original/native maps include fixed removal and effective RelaxBounds. Begin events precede factor/backsolve invocation; End includes failed calls even after a terminal abort. An observer abort is sticky and stops numerical fallback. Scope destruction restores the previous worker-local state.

Storage scope Application names the deployment-owned opaque TNLP/algorithm heap. Scope Linear names owner allocation reservations; no actual opaque heap count is asserted. Reservations must remain retained through destruction of application/factor owners, including numerical fallback while Schur owners still live.

## Bounded linear profile

`FeralConfig::bounded_dense_max_dimension=Some(1..=7)` selects the serial AMD tiny dense profile with Auto/InfNorm/Identity scaling. Assembled IPM curvature is required by the consumer. Unsupported geometry or configuration is refused; it does not silently run the unrestricted profile. Larger unrestricted profiles retain ordinary behavior and emit opaque Linear requests.

The profile reuses `dense_fast_factor_with_workspace` and its existing scalar kernel. It omits the generic driver's unused symbolic analysis, race, parallel pool and MC64 retry policy. It preserves the dense numeric arithmetic, scaling, factor and inertia representation; a rank-deficient factor remains a rank-deficient outcome for the calling linear owner.

The complete requested-allocation bound is source-derived and conservative, not a measured allocator extent. FERAL's bound inventories six simultaneous n-square dense/contribution buffers, old/new factor vectors, scaling and norm scratch, scalar scratch and up to seven RHS of solve/refinement work. Factor replacement is included. The factor interface adds retained CSC/slot/index/value arrays and CSC construction/debug rebuild scratch. The Schur bound adds both factors, old/new block matrices, all split/maps, AFS/W/ASTW/ASS/S dense storage, triplet packing and transient back-substitution results. The assembly owner separately admits its retained/replacement triplets, copies, flattened vectors and packed RHS. Pre-builder admission includes main/restoration/fallback wrapper headers. Vector growth uses twice logical maxima plus minimum-four allocations. The source functions document the individual dimensions and counts.

The bound covers requested buffer extents and their owning objects; platform allocator metadata, tracing and NLP algorithm allocations remain deployment-owned. Refinement uses explicit `RefineOptions` caps. Primitive observations count public FERAL factor and backsolve invocations; unrestricted internal retries and actual refinement steps remain unknown.

Tests exercise actual bounded POUNCE Schur execution, terminal abort without fallback, layout binding and bitwise numeric parity across all supported dimensions and1..7 RHS.
