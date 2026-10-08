# VPA (Vision-Predictor-Action) — PyTorch reference implementation

PyTorch implementation of the Goal-Conditioned VPA framework described in `docs/preprint_261008.pdf`.
**That PDF is the only ground truth** for every formula, tensor dimension and data dependency.
If the code and the PDF disagree, the PDF wins. Read the relevant section of the PDF before you
change any math.

## Layout

| File | Paper | Contents |
|---|---|---|
| `perception.py` | Sec. 4.1, Eq. 7 | `VisionEncoder` (Siamese ViT E_ψ), `MomentumEncoder` (EMA target E_ψ̄), `TextEncoderWrapper` (frozen CLIP → c_text) |
| `selector.py` | Sec. 4.2, Eqs. 8, 9a–9c | `NeuroSymbolicSelector` (MLP π_φ^h), `MilestoneTracker` (pointer m_t, τ = κ√(2d)γ̄) |
| `predictor.py` | Secs. 4.3, 5.2, Eqs. 10, 11, 18, 19 | `JEPAPredictor` (N_e-head ensemble, σ_{t+1}), `VICRegLoss` |
| `solver.py` | Secs. 4.4, 4.5, Eqs. 12–15 | `FlowMatchingSolver` (v_θ, Euler K-step `generate_chunk`), `standardize_latents`, `UncertaintyHorizonFilter` |
| `pipeline.py` | Fig. 2, Prop. 5.1 | `VPAConfig`, `VPAInferencePipeline` (`reset`, `step`, `act`) |
| `docs/preprint_261008.pdf` | — | the paper (ground truth) |

## Environment (Mac mini M4, Apple Silicon)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers ruff        # macOS arm64 wheels include the MPS backend
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

- Python ≥ 3.10 is required (`zip(..., strict=True)` is used).
- **Run the self-tests on CPU.** They are deterministic and use float64 for reference values.
  Use MPS (`device="mps"`) only for pipeline smoke and latency runs.
- **MPS has no float64.** Never put float64 tensors in module parameters or buffers, or anywhere
  on a forward path. float64 is allowed only in CPU-side reference computations inside tests.
- `torch.Generator` objects are bound to a device. For tensors on MPS, use
  `torch.Generator(device="mps")`.
- If an op is not implemented on MPS, run with `PYTORCH_ENABLE_MPS_FALLBACK=1`. Mention it in
  your report when you do. Don't rewrite the math to avoid the op.
- When timing MPS, call `torch.mps.synchronize()` before reading the clock.
- `TextEncoderWrapper()` downloads `openai/clip-vit-base-patch32` from Hugging Face on first use.
  The self-tests use an offline stand-in, so they need no network.

## Commands

```bash
# All self-tests, in dependency order (pipeline last); stop at the first failure
for f in perception selector predictor solver pipeline; do python "$f.py" || break; done

# One module
python solver.py

# Lint (must pass)
ruff check --select F,E9,B,PLE,PLW .
```

## Current status

Verified on macOS 15 / Apple M4 with Python 3.13, torch 2.14.1 and transformers 5.19.0
(`requirements.txt`). All networks still have random weights; there are no trained checkpoints.

Done:

1. All five self-tests pass on CPU in the order above, and the lint passes. The one failure found
   on the first run was in a test helper (`solver._CountingField` recorded ρ_k in float32).
2. Pipeline smoke test on MPS: `VPAConfig()` → `from_config` (real CLIP) → `calibrate` → `reset` →
   `act`/`step` reports `num_sequential_evaluations == K + 3` (the README quick start).
   `bench_latency.py` also asserts K + 3 on every `step()` for K ∈ {1,2,3}, H ∈ {8,…,128} on MPS.
3. Every module's `_self_test()` checks values against independent float64 references, covering
   the edge cases listed in the testing rules below.
4. An independent equation-by-equation audit against the PDF found no discrepancies. The
   interpretive choices it surfaced are listed under "Decisions where the paper is silent".

Open (see the README roadmap): training loops and checkpoints, the Section 6 evaluation
(Isaac Sim, ManiSkill3), the per-milestone time-out and spectral normalization.

## Non-negotiable rules

1. **Mathematical strictness.** Every tensor operation implements an equation from the PDF.
   Approximations and alternative algorithms are not allowed. Cite the equation number in a
   comment next to the code.
2. **Shape documentation.** Every class and every `forward` has type hints and a docstring with
   input and output shapes, written like `[B, H, d_a]`.
3. **K + 3 sequential depth.** `VPAInferencePipeline.step` must evaluate exactly
   `[E_psi, pi_phi_h, P_omega] + [v_theta] * K`. Forward hooks audit this sequence at runtime,
   and a mismatch raises. Never remove or weaken that check. Always call networks through
   `module(...)`, never `module.forward(...)`, because direct `forward` calls bypass the hooks.
   No hidden loops whose length depends on H.
4. **Module independence.** `perception`, `selector`, `predictor` and `solver` import nothing
   from each other. Only `pipeline.py` composes them. Each file keeps a `_self_test()` under
   `if __name__ == "__main__":`.
5. **No stubs.** No `pass`, `TODO` or placeholder bodies in core logic (ODE loop, VICReg terms,
   σ, thresholds, H_t).

## Exact forms used (check changes against the PDF)

- ‖·‖₂² sums over the feature dimensions; 𝔼 is the batch mean. `Var` is the **unbiased**
  (B−1) estimator, the same estimator as C(Z).
- Eq. 8: `u_t = argmax π_φ^h(u | z_t, z_g^(m_t), c_text)`; the MLP input is `concat(z_t, z_g, c_text)`.
- Eq. 9a: `m_{t+1} = m_t + 1` if `‖z_t − z_g^(m_t)‖₂ < τ` (strict) and `m_t < M`. Task complete
  when the test holds at `m_t = M`.
- Eq. 9b: `γ̄_n = (1−α_γ) γ̄_{n−1} + α_γ · sqrt( (1/d) Σ_j Var(Z_{·,j}) + ε )`, using online-encoder latents.
- Eq. 9c: `τ = κ √(2d) γ̄`; deployment uses `τ* = κ √(2d) γ̄*`.
- σ (Sec. 4.3): `σ = sqrt( (1/N_e) Σ_i ‖ẑ^(i) − ẑ‖₂² ) / (√d · γ̄*)`, where ẑ is the ensemble mean.
- Eq. 11: `L = 𝔼‖ẑ − sg(E_ψ̄(I_{t+1}))‖₂² + λ_v v(Z) + λ_c c(Z)`.
- Eq. 18: `v(Z) = (1/d) Σ_j max(0, γ − sqrt(Var(Z_{·,j}) + ε))`, with `0 < ε < γ²`.
- Eq. 19: `c(Z) = (1/d) Σ_{i≠j} C_ij²`, where `C = (1/(B−1)) Σ_b (Z_b − Z̄)(Z_b − Z̄)ᵀ`.
- Eq. 12: `e_t = concat(Embed(u_t), z̃_t, z̃̂_{t+1}, S̃_t, c_text)`, `d_e = d_u + 2d + d_s + d_c`.
  Sec. 4.5: `z̃ = (z − μ̄_Z)/γ̄*`, `S̃ = (S − μ_S)/σ_S`.
- Eqs. 13–14: `A^(ρ) = ρA + (1−ρ)ξ`, the target velocity is `A − ξ`, and `ρ ~ U[0,1]`.
- Euler (Sec. 4.4): `A^(0) = ξ`, `A ← A + (1/K) v_θ(A, ρ_k | e_t)`, `ρ_k = k/K`, with `K ∈ {1,2,3}`.
- Eq. 15: `σ̄ = max{σ, (1−α_σ)σ̄_prev + α_σ σ}`, `H_t = max(H_min, ⌊H_max · exp(−β σ̄)⌋)`.
- Constraint: predictor stride `ν ≤ H_min`.

## Decisions where the paper is silent

Don't change these silently. If you change one, update the docstring and say so in your report.

- γ̄₀ = the first instantaneous estimate (unless `gamma_bar_init` is given).
- The invariance term is averaged over ensemble heads (`head_reduction="mean"`; `"sum"` is available).
- Selection at step t uses m_t; m_{t+1} applies from the next decision step.
- σ̄ starts at 0 each episode, so the first step has σ̄ = σ.
- c_text is not standardized; Sec. 4.5 only names the latents and S_t.
- `VPAConfig` defaults and the EMA momentum (0.996) are placeholders, not values from the paper.
- The per-milestone time-out (deferred in Sec. 4.2/6) and spectral normalization (optional,
  Sec. 5.3) are intentionally not implemented.

Pipeline and episode semantics:

- The pointer starts at m₀ = 1. At most one milestone advances per decision step; the Eq. 9a test
  runs only at decision steps, never during chunk execution.
- Order inside `step`: Eq. 15 filter, then Eq. 9a, both on the same z_t, after the K + 3 chain.
  Nothing is committed if the depth audit fails.
- Completion only sets a sticky `task_complete` flag. The step that completes still generates and
  commits a chunk, and `act` keeps returning actions; ending the episode is the caller's job.
- Batches replan asynchronously: the networks run on the whole batch, but σ̄, m_t, the chunk, H_t
  and the cursor are committed only where `decision_mask` is set. The first step must include every
  element. (The paper describes a single agent.)
- Ties in Eq. 8 go to the lowest primitive index (`torch.argmax`).
- `calibrate` freezes only E_ψ and P_ω (Sec. 4.3); selector and solver stay trainable but are put
  in eval mode.
- ν is validated (1 ≤ ν ≤ H_min) but unused at inference; it only matters for aligning training
  targets.
- The ε of Eq. 9b and Eq. 18 must be the same value. `VPAConfig.vicreg_eps` sets the tracker's ε;
  build the loss with `VPAInferencePipeline.make_vicreg_loss`, or verify an external one with
  `check_vicreg_loss`. (`VICRegLoss` can't read `VPAConfig` itself because of module independence.)

Numerics and estimators:

- The ensemble mean ẑ is computed as `ẑ⁽¹⁾ + mean_i(ẑ⁽ⁱ⁾ − ẑ⁽¹⁾)`: the same mean in exact arithmetic,
  but σ is exactly 0 when all heads agree.
- Eq. 14 is a one-sample Monte Carlo estimate per batch element: one ξ and one ρ per call, with
  ρ drawn from `torch.rand`, i.e. [0, 1).
- S̃ uses a per-dimension σ_S vector; latents use the scalar γ̄* and a per-dimension μ̄_Z.
- float32 everywhere on the forward path (MPS). So ρ_k = 1/3 is rounded inside v_θ, γ̄ is computed in
  float32 even for float64 input, and the H_t floor can differ from exact arithmetic by one exactly
  at an integer boundary.

Architectures the paper leaves open:

- E_ψ: pre-norm ViT; the latent is a linear head on the final LayerNorm'd [CLS] token.
- E_ψ̄: starts as an exact copy of ψ; buffers are copied, not EMA-averaged (the ViT has none).
- c_text: CLIP's projected `text_embeds`, not L2-normalized, padded and truncated at 77 tokens.
- P_ω: N_e GELU-MLP heads, each with its own primitive embedding (N(0,1) init, separate from the
  solver's Embed in Eq. 12), weights U(±1/√in).
- v_θ: non-causal pre-norm transformer over the H positions, with a prepended conditioning token
  plus an additive broadcast of MLP(e_t, time(ρ)); sinusoidal ρ features scaled by 1000.

## Conventions

- Images are channels-first: `[B, C, H_img, W_img]`. Milestones are `[B, M, C, H_img, W_img]`.
- The milestone pointer `m_t` is 1-based, as in the paper. It is converted to 0-based only for `gather`.
- γ̄* is installed once through `VPAInferencePipeline.calibrate`, which sets the tracker, the
  predictor and the solver together. Don't set them individually.
- Training loops are out of scope unless asked. The losses (`VICRegLoss`, `flow_matching_loss`,
  `NeuroSymbolicSelector.loss`) already exist.

## Rules for tensor-calculation self-tests

- Compute the **reference independently** from the PDF formula, with explicit loops or naive
  algebra in float64 on CPU. Never reuse the function under test to build its own reference.
- Fix seeds (`torch.manual_seed`, explicit `torch.Generator`). Test values, not only shapes.
- Tolerances: compare float32 module output against the float64 reference with
  `rtol=1e-5, atol=1e-6`. Loosen a tolerance only with a stated numerical reason.
- If a test fails, check the equation in the PDF first, then decide whether the module or the
  test is wrong. Never edit a reference computation just to make a test pass.
- Cover these edge cases: the collapsed batch from Prop. 5.2(i) (`v = γ − √ε`, `c = 0`);
  `B < 2` raises; every `K ∈ {1,2,3}`; depth stays `K + 3` for different H; pointer
  monotonicity and `m_t = M` completion; `H_t` clamps at `H_min`; masked (per-element)
  replanning in `act`.
- Keep each test inside its own module's `_self_test()`, so every file can still be tested on
  its own.
