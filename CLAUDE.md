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

The code was written in an environment without PyTorch and **has never been executed**.
It compiles and passes the lint above. Work in this order:

1. Run each self-test on CPU in the order above and fix the failures.
2. Run a pipeline smoke test on MPS: build with `VPAConfig()`, `calibrate(...)`, `reset(...)`,
   then `step(...)` and `act(...)`, and check that `num_sequential_evaluations == K + 3`.
3. Add or extend the tensor-calculation self-tests (see the testing rules below).

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
