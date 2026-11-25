LANGUAGE_CARD = """WSL v0.1 - Weight-Space Language. One statement per line:  [LABEL:] CLAIM OP; OP; ...   (# starts a comment)
Directives: "model NAME" (neox-tiny | pythia-14m | modadd | grid:taskNNN) and "seed N". Every statement starts from
the unedited model and is compared with it. Layers: L[i] or Li (0-based), L[mid], L[mid-1], L[mid+1]; heads L[i].H[h].

CLAIMS
  gauge      the pipeline is an exact symmetry (logits unchanged up to float64 round-off)  -> CONFIRMED / REFUTED
  reparam    function-preserving change of parametrisation (may add parameters)           -> CONFIRMED / REFUTED
  approx(t)  relative logit change <= t                                                     -> CONFIRMED / REFUTED
  edit       changes the function; measured as delta-loss, with a random control of equal norm -> MEASURED
  train      train a modular-addition model (grokking toy)                                  -> MEASURED
  path       evaluate points between / along checkpoints                                     -> MEASURED
  solve      an ARC grid program or compiled network must pass every example of the task     -> CONFIRMED / REFUTED

OPS ON GPT-NeoX MODELS (neox-tiny, pythia-14m: LayerNorm, GELU MLP, parallel residual, partial RoPE)
  fold_ln()                         move every LayerNorm gain/bias into the next matrix (adds embed_out.bias)
  center_writers()                  remove the all-ones component from everything written to the residual stream
  ln_to_rms()                       replace LayerNorm by RMSNorm (exact only after fold_ln(); center_writers())
  rotate(resid, orth|orth1, layers=all|[i,...], embed=true|false, unembed=true|false)
                                    x -> R x on the residual stream; orth1 means R fixes the all-ones vector
  scale(resid, c)                   multiply the embedding and all writers by c
  permute(L[i].mlp, randperm|perm([...]))      permute MLP neurons      scale(L[i].mlp, uniform(lo, hi))   rotate(L[i].mlp, orth)
  permute(L[i].heads, randperm|perm([...]))    permute attention heads
  vo(L[i].H[h], gl|orth)            V -> N V, O -> O N^-1 for one head
  qk(L[i].H[h], gl|orth, dims=pass|all)        Q -> A Q, K -> A^-T K on the non-rotary dims (pass) or all dims
  rope_pairs(L[i].H[h], phis=[...], scale=s)   same 2x2 rotation on each RoPE pair of Q and K; Q*s, K/s
  delete(L[i])   swap(L[i], L[j])   L[i] := L[j] | lerp(X, Y, t) | neg_writers(X) | neg(X) | align(X -> Y)
  gridop(PRIM, block("tensor name", r0, r1, c0, c1))    apply an ARC grid primitive to a block of a weight matrix
  PRIM: identity rot90 rot180 rot270 hmirror vmirror dmirror cmirror  (rot90 is clockwise; hmirror flips top/bottom)

OPS ON modadd
  train modadd(run=NAME, p=97, frac=0.5, steps=3000, seed=0, lr=0.001, wd=1.0, filter=none|grokfast(alpha, lamb),
               save=[steps], intervene=[at(STEP, scale(C))])
  path lerp(ckpt(R, s1), ckpt(R, s2), t=[a:b:step])     path add(ckpt(R, s), diff(ckpt(A, s2), ckpt(A, s1)), t=[a:b:step])
  path scale(ckpt(R, s), c=[a:b:step])                  path cosine(diff(...), diff(...))

OPS ON grid:taskNNN (ARC-AGI-1 task from the 2025 Code Golf set)
  solve prim(PRIM)    solve chain(prim(A), prim(B), ...)    solve golf()
  solve compile(PRIM, size=fixed|gated, from=all|train)   one-hot (10,30,30) permutation network; 'gated' = one per input size

FACTS MEASURED ON THESE MODELS (float64; exact = relative logit change about 1e-12)
  exact:  MLP neuron permutation; head permutation; V/O change of basis per head; Q/K change of basis on non-rotary dims;
          one 2x2 rotation per RoPE pair shared by Q and K (plus Q*s, K/s); fold_ln; center_writers; after those two,
          ln_to_rms; residual rotation of ALL readers, writers, embedding and unembedding after fold_ln with R fixing
          the all-ones vector (orth1), or with any R after center_writers + ln_to_rms; aligning a layer to another.
  not exact: residual rotation without fold_ln; with LayerNorm and R not fixing ones; of one layer only (a large edit,
          worse than deleting the layer); per-neuron positive rescaling through GELU; rotating MLP neurons; Q/K basis
          change on rotary dims; scaling the residual stream (only LayerNorm eps breaks it: approx).
  pythia-14m edits (delta-loss in nats, base 4.3177): delete middle layer +4.04; swap layers 2,3 +3.27;
          L4 := -L2 (writers negated) +2.51 (delete L4 alone +2.16, delete L2 and L4 +4.06); L3 := mean(L2, L4) +4.99.
  modadd (p=97, 50 % train): memorised by step ~150, generalises (test >= 0.9) by step ~1200; Grokfast ~1.2x earlier;
          the straight line memorised -> grokked has a barrier (train acc ~0.8 at t=0.25); the direction of one seed
          is orthogonal to another seed's (cosine ~0) and transplanting it destroys the other model."""
