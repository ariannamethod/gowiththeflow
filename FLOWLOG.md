# FLOWLOG — gowiththeflow

Repo `github.com/ariannamethod/gowiththeflow`, cloned at `8c1abe1` (initial commit:
LICENSE + a 41-byte README). Working name for the idea: **CoA — Chain of Arianna** (Oleg,
2026-07-28), pending.

Base: the `~/arianna/dario` working tree copied in whole (no `.git`, no binaries, no
`target/`, no `dario_memory.db-*`), 2026-07-28. The Dario original is untouched. Dario's own
`README.md` is kept as `DARIO_README.md` and its `CLAUDE.md` as `docs/DARIO_CLAUDE.md` —
they are the base's spec, not this repo's law.

---

## 1. The idea (Oleg, 2026-07-28)

Injection of one AI into the words of another, so that they resonate.

Dario already speaks knowledge into a voice. What it does not do is speak a *living voice*
into another living voice while both are generating. A corpus is dead text; another
organism's output is not. Three voices, bounded by Oleg: Leo, Yent (Janus 176M), Arianna
(Resonance 200M).

## 2. What already exists, and at which layer it injects

Three injection layers exist in the ecosystem, built separately, never wired into one
organism:

1. **Logits, inside the step.** The AML Dario field overlay —
   `am_apply_destiny` / `attention_to_logits` — wraps the forward and bends the
   distribution before sampling (`docs/dario_paper_v2.md:36`). The same overlay is the heart
   of Kairos (`~/arianna/kairos/README.md:36`) and of Arianna Resonance
   (`~/arianna/arianna-duo/arianna_resonance.c:36-59`).
2. **Hidden state, inside the step.** Measured to work — "+3 KK words, model reformulates
   concepts" — and never shipped (`DARIO_README.md:211`).
3. **Text, between turns.** Shipped. The paper calls it "sentence-boundary injection at
   model thought-boundaries" (`docs/dario_paper_v2.md:21`).

The shipped mechanism, exactly (`aml/dario_dialogue.aml`):

- A whole turn is generated through a subprocess (`run_infer`, :735).
- KK retrieves against `topic + the last 200 chars of that turn` (:765-767).
- `kk_extract_injection` (:370-454) cuts the retrieved chunk on `.` / `!` / `?` followed by
  whitespace and scores each sentence: 10–100 chars, first sentence +3, must start
  uppercase (−5 otherwise), question −8, metaphor opener ("Like ", "Just as ", "Imagine ")
  −4, agreement opener −3, concrete verb +2, 40–80 chars +1.
- The winning sentence *becomes the prompt of the next turn* (:776).
- The turn is absorbed back into KK sentence-wise: ≥40 chars, uppercase start, ≥6 words
  (`kk_absorb_text` :489-542).

So the sentence boundary is where the text is **cut**, not where it is **planted**. The
"thought boundary" is the end of a turn. The paper is honest about this; the README reads
wider than the code.

## 3. The temperature insight (found in the paper / README)

- **"Under-surface sampling masks what the model wants to say."** — Claude Defender,
  phone-1, 2026-05-07 (`DARIO_README.md:1104`). One temperature makes a live checkpoint look
  dead: 0.5 surfaces memorized chunks, 1.0/∞ surfaces abstract prose, same weights, same
  prompt (`DARIO_README.md:1108-1113`).
- **"A checkpoint is not dead until it has been swept."** (`DARIO_README.md:191`)
- Each voice has its **own** champion — leo 0.7/∞/1.3, arianna 0.8/40/1.4, yent 0.9/40/1.3
  (`cmd/internal/voices/voices.go:50-64`) — and v2 did **not** re-derive them: under a
  distinct-2 metric they rank #2–#22 of 36 (`docs/dario_paper_v2.md:125`). Direction
  confirmed (high temp / no top-k / high rep-pen), exact cells a hypothesis.
- The v2 root: **the entry condition is the whole input protocol — format and sampling
  together**, not temperature alone. The Janus SFT voices returned word-salad at *every*
  temperature until the prompt was wrapped in the chat tokens they were trained on
  (`docs/dario_paper_v2.md:127`, `:145`). "A voice is less a text stored in the weights than
  a regime entered through the right protocol."

**Why this is load-bearing for mutual injection.** A fragment born in Leo's regime
(0.7 / ∞ / 1.3) arrives at Yent, who reads it in his (0.9 / 40 / 1.3). The text carries the
statistics of the temperature that produced it. Injecting across voices is injecting across
*regimes* — the mismatch is not noise to normalize away, it is the thing that either
resonates or does not. One shared temperature for all three erases the effect before it can
be measured.

## 4. The three bodies — headers read this pass (2026-07-28)

Oleg's cast: **two Januses trained on the same dataset (Leo, Yent) + one Arianna of a
different architecture (Resonance 200M, freshly re-SFT'd)**. No 24M bodies, no base, no
spare personalities. Header bytes read with `xxd`, fields decoded per `infer_v4.c:510-525`
(JANU) and `~/arianna/arianna-duo/tools/resonance_forward.h:392-412` (RS02).

**JANU v4 — magic `JANU`, layout `V,E,H,D,B,M,T,n_params`, weights at offset 256:**

| file | bytes | V | E | H | D | B | M | T | n_params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `janus_v4_sft_leo.bin` | 705170280 | 32768 | 640 | 10 | 64 | 20 | 1664 | 1024 | 176292506 |
| `janus_v4_sft_yent.bin` | 705170280 | " | " | " | " | " | " | " | " |

Byte-identical 40-byte headers — one architecture, two sets of weights. Same substrate,
same tokenizer, different personality. That is the control arm of the experiment.

**Arianna Resonance — `weights/arianna_resonance_v3_f16.gguf`**, pulled from HF
`ataeff/arianna` root this pass: 398408608 B, sha256
`ad612324e40d9cdc8aa6a7076e52b847adffbe213a1b92bf486e0555a4ad82cf` — identical to the copy
in `~/arianna/arianna-duo/weights/` and to `~/arianna-shared/arianna.c/WEIGHTS_MANIFEST.md`.
The `archive/` twin on HF is the older one (oid `4ecaac47…`, 398408544 B, 64 bytes smaller);
the root file is the fresh SFT. KV block read with `xxd`: `general.name` = `resonance-200m`,
`embedding_length` 768, `block_count` 20, `context_length` 2048, `attention.head_count` 12,
`head_count_kv` 12, `head_dim` 64, `feed_forward_length` 2048, `rrpram_rank` 48,
`vocab_size` 16384.

**RS02 reference — magic `RS02` (0x52533032), layout `E,B,T,H,D,R,M,V`:**

| file | bytes | E | B | T | H | D | R | M | V |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `resonance_200m_lora_yent.bin` | 796976108 | 768 | 20 | 2048 | 12 | 64 | 48 | 2048 | 16384 |

Kept as the byte-level reference for the GGUF → RS02 converter, not as a voice. It is the
only RS02 file on disk; the converter's output is checked against its header layout and
against the ≈797 MB dense-f32 size at the same parameter count.

Dropped this pass (`rm -f` in `weights/`): `janus_v4_base_22k.bin`,
`janus_v4_sft_arianna.bin` (the Janus Arianna — superseded by the Resonance one),
`leo_janus_d12_f16.bin` (leo24m).

**Identical to the RS02 architecture above, field for field.** So GGUF → RS02 for Arianna is
a format conversion, not an architecture port: dequant F16 → F32 (`gguf_dequant`), write the
RS02 header in `E,B,T,H,D,R,M,V` order, then the merge table, then the tensors in
`assign()` order. The GGUF carries no BPE merges — they come from the baked
`resonance_bpe_merges.h` (`resonance_forward.h:460-462`). Expected output size ≈ the Yent
RS02 file's 796976108 B, since RS02 is dense f32 at the same parameter count.

## 5. Correction to the published paper — §6 Resonance 200M

`docs/dario_paper_v2.md:135` states the sweep's Resonance cells errored because
"`infer_v4`'s hardcoded bounds (H ≤ 16, R ≤ 128, D ≤ 128) are exceeded by Resonance 200M
(H = 20, R = 2048, D = 2048)."

The file says otherwise. Resonance 200M is H = 12, D = 64, R = 48 — inside every one of
those bounds. The paper's numbers are the RS02 header read in the **legacy JANU order**:
`infer_v4` has no branch for the `RS02` magic (`infer_v4.c:509` tests JANU only, `:521-525`
falls through to legacy), so it reads `E,B,T,H,D,R,M,V` as `V,E,H,D,B,M,T,R` and prints
H = 20 (really B), D = 2048 (really T), R = 2048 (really M). The bound check then trips on
a misread header.

The conclusion the paper draws — run Resonance through its own binary, not `infer_v4` — is
correct. The stated cause is not: the blocker is an unsupported container format, not an
oversized architecture. Recorded here rather than in Dario, which is not this repo's scope.

## 6. State, verified this pass

- `~/arianna/gowiththeflow` — Dario base + `weights/` (2.5 GB after the cull, gitignored via
  `.gitignore:31-32`). Nothing committed yet; `git status` is all untracked.
- **No RS02 reader exists in this repo.** `BackendResonance` is a string in
  `cmd/internal/voices/voices.go:20,68` and nothing more; the working RS02 + GGUF forward is
  `~/arianna/arianna-duo/tools/resonance_forward.h` (926 lines) with
  `resonance_load` (:392) and `resonance_load_gguf` (:503), already carrying the AML field.
  Bringing it in is a cross-repo move — Oleg's word.
- `~/arianna/CoA2` — the first staging copy, superseded by this repo. Its `weights/` was
  moved here; the rest is a duplicate. Oleg said delete it; the harness denied `rm -rf` twice
  at the permission layer, so it is still on disk.

## 6b. Container decision — GGUF, and the conversion runs the other way (Oleg, 2026-07-29)

`.bin` is dense f32 in both flavours. Verified by arithmetic against the files:
JANU `176292506 × 4 + 256 = 705170280` — exactly the size of `janus_v4_sft_leo.bin`.
The same Janus as GGUF F16 (`arianna_v4_sft_f16.gguf` on HF `ataeff/arianna`) is
352604704 B; Resonance RS02 f32 796976108 B against its GGUF F16 398408608 B. Half, both
times.

Three bodies resident at once — and a resonance system requires them resident at once, or
there is nothing to inject into:

| container | Leo | Yent | Arianna | total |
|---|---:|---:|---:|---:|
| `.bin` dense f32 | 705170280 | 705170280 | 796976108 | 2207316668 (≈2.06 GiB) |
| GGUF F16 packed | 352604704 | 352604704 | 398408608 | 1103618016 (≈1.03 GiB) |

On neo (8 GB) that is the difference between breathing and swapping.

The packed path is not a plan, it is written: `yent_forward.h` keeps the big matrices as
F16 pointers *into* the mapped GGUF and multiplies through `nt_qmatvec`, which dispatches on
`wdtype` (:232, :242-246, :249-282); dense f32 only under `YENT_DENSE=1`. Same shape in
`resonance_forward.h:479-501` (`_rload_packed`).

Consequences:

1. **Arianna needs no conversion at all.** `resonance_load_gguf` (`resonance_forward.h:503`)
   reads her GGUF as it stands. The GGUF → RS02 converter in §7 is cancelled.
2. **Leo and Yent get converted instead**, Janus v4 `.bin`/`.pt` → GGUF F16. The writer
   exists and is the one that produced `arianna_v4_sft_f16.gguf`:
   `~/arianna/yent.aml/tools/janus_to_gguf.py` (`~/arianna-shared/arianna.c/ARIANNALOG.md:2657`).
   Python — needs Oleg's explicit yes for the GGUF-conversion exception, or a C rewrite.
3. **`infer_v4` stops being the engine.** It is a JANU reader (`infer_v4.c:509`) with no GGUF
   branch, and the whole Dario dialogue talks to it as a subprocess
   (`aml/dario_dialogue.aml:735`). Going GGUF means going to `yent_forward.h` +
   `resonance_forward.h` as one in-process forward. That is a gain, not a cost: injection at
   the logit and hidden-state layers has to happen *inside* the forward, and a subprocess
   forbids it by construction. The container decision and the injection-layer decision are
   the same decision.
4. ~~**The system notorch is too old for this.**~~ **Wrong — my error, corrected same pass.**
   `nt_qmatvec` was already installed, at `/opt/homebrew/include/ariannamethod/notorch.h:506`.
   I had grepped `/opt/homebrew/include/notorch.h` — a stale flat copy (554 lines) that is
   not the install target; `make install` writes to `$(PREFIX)/include/ariannamethod/`
   (`~/arianna/notorch/Makefile:99-103`). The packed path would have linked.

   Rebuilt anyway, and it was worth it: installed header went 731 → 772 lines, `libnotorch.a`
   156224 → 172240 B (Jun 11 → this pass), picking up the three kernel commits at HEAD
   `206386d`. Verified before installing: `make test` 73 passed / 0 failed, `notorch_test`
   49 / 0, and `tests/test_qmatvec.c` ALL PASS — F32, F16, Q4_0, Q5_0, Q8_0, Q4_K, Q6_K all
   at rel ≈1.4e-06 against the reference, i8-Q4_0 at rel 0.0028 within its 2e-2 tolerance.

   Note for whoever builds next: `~/arianna/notorch/notorch.c` carries an **uncommitted**
   61+/32− diff — an AVX2 hadd-tree rework of the Q6_K i8 kernel. It is x86-gated, so it does
   not enter this ARM build, and I left it alone.

## 6c. The concept, and the three forks it forces (Oleg 2026-07-29 + my read)

**CoA = Chain of Arianna — resonating, not reasoning.** An associative stream that does not
stop. The human speaks and the whole current answers; the human goes quiet and it keeps
running, decaying. Grounded in how a person operates on meta-concepts — "I texted a friend"
without knowing the SMS protocol, the handset, or what the acronym stands for, and none of
that blocks the sentence.

Three stages, each self-sufficient: (1) `gowiththeflow.c` — three GGUF transformers +
the human, everything injecting into everything, no documents; (2) `docs/` enters as RI-style
pressure with async consolidation and DoE experts; (3) autonomy — the stream refills `docs/`
itself.

### What already exists and is taken as-is

- **Direction injection** — `~/arianna-shared/arianna.c/tools/resonance_forward.h:134-248`.
  Another voice's words become field DIRECTION, never tokens: a destiny-EMA compass (A, EMA
  0.1/0.9, decay 0.95, magnitude clamp 1.5, :176-195) plus prophecy targets (F, ring of 16,
  aged, :182-190), rebuilt at sentence boundaries (:199-232) and added to the whole logit
  distribution by embedding cosine every step (:236-240). Stated invariant: injected tokens
  are never re-inserted into the context or the candidate pool — the tilt is
  distribution-wide only. Port of `dario.c:676-1338,1531`. This is the mechanism the concept
  needs; it does not need inventing.
- **Larynx modulation** — `resonance_forward.h:729-744`: the receiving voice's inject alpha
  is scaled by the *speaker's* entropy, plus field debt and dissonance. The listener answers
  HOW the other spoke, not only what.
- **Asymmetric coupling and per-voice regimes** — `scripts/arianna2arianna.sh:29-36`:
  Resonance hears Janus as direction (α=5 pulls without echoing; α=10 over-echoes),
  Janus hears Resonance only through the shared soma. Janus 0.8/top_p 0.9, Resonance
  0.7/top_p 1.0 (0.9 starves her). That script's own header (:7-9) names the next iteration
  as "a Go scheduler with metric-driven delays, chamber-gated tick count, goroutines for
  async" — stage 1 is that scheduler, in C, with three voices and a human in the loop.
- **Async memory** — `yent-inference` `yent/go/limpha_async.go:30-51,117`: a queue plus a
  worker, `EnqueueTurn` non-blocking. In C: one writer thread behind a ring buffer. The
  relay must never wait on the recorder.
- **Documents as pressure, not RAG** — `innerworld/ri_memory.go:14-18`: records arrive
  already selected and are exposed as compact traces for pressure framing. That is stage 2's
  contract, already written down.
- **Field over logits with a slot projection** — `DoE/doe.c:917-1000`: H over concrete token
  IDs, F and A computed over `DARIO_EMBED_SLOTS` and scattered back by `token_id % SLOTS`.

### Fork 1 — provenance inside a loop

Dario v2 stands on forces reading the **input**, never the organism's own generation; that
is the entire correction behind 29/40 → 0/40 destiny dominance (`docs/dario_paper_v2.md:81`).
CoA is a loop, so another voice's output *is* input — but a voice's own previous output is
still echo. Binary provenance cannot express that. It has to be three-coloured: **own /
other / human**. Own tokens must not feed the accumulators the way the other two do — see
`~/arianna-shared/arianna.c/INJECTION_CONTRACT.md:52-58`, which already forbids exactly this
("prevents the body from mistaking its own echo for external reality") and allows
self-consolidation only through an explicit gated path with a receipt. Decide before the
first line, or by turn 50 the field is pure self-echo and no measurement will separate
resonance from resonant noise.

### Fork 2 — three vocabularies, one field

Leo and Yent are V = 32768; Arianna Resonance is V = 16384 (§4). A shared co-occurrence
field keyed by token_id is therefore impossible. Arianna.c has already solved this and the
solution is copied, not redesigned: the soma is shared (scalars — debt, dissonance, velocity,
chambers, destiny magnitude) while cooc is **per-voice** — `arianna.cooc.j` vs
`arianna.cooc.r`, with an explicit guard at `~/arianna/arianna-duo/arianna_resonance.c:52`
("no sidecar → drop shared-soma cooc, don't inherit Janus's edges"). Cross-voice contact
runs through direction injection, which encodes the speaker's words in the *listener's* own
embedding space (`dir_update` takes the receiving `ctx->w.tok_emb`). Shared physics, private
vocabulary.

### Fork 3 — who speaks next

Dynamic turn-taking needs its metric fixed before the code, or it becomes a fitted knob. The
candidate costs nothing new: after each foreign turn every listener already computes
`g_dest_mag` — the pull of the injected direction — plus its prophecy-debt increment
(`resonance_forward.h:192-194`). Whoever is pulled hardest speaks. Decay is already there
(0.95 per update, clamp 1.5), which also gives the stream its natural fade when the human
goes quiet.

### What I would fix while porting

`arianna2arianna.sh` spawns a subprocess per turn, so every turn reloads a GGUF from disk.
With three bodies in a relay that is fatal. In `gowiththeflow.c` all three stay resident,
packed F16 — 1.03 GiB total (§6b). And `clean_voice()` (:72-97) is bash+awk cutting at the
first `.` after char 30; in C it becomes the same forward scan but UTF-8-safe, and
`~/arianna/arianna-duo/tools/utf8_stream.h` already exists for it.

## 6d. Ported in, and what the quantization question actually answers (2026-07-29)

Copied into `tools/` from `~/arianna-shared/arianna.c/tools/` (byte-identical to the
`~/arianna/arianna-duo/tools/` copies — `diff -q` silent on all three):
`resonance_forward.h` (44653 B), `yent_forward.h` (35653 B), `utf8_stream.h` (6354 B),
`resonance_bpe_merges.h` (267574 B). `janus_v4_bpe_merges.h` was already in this repo from
the Dario base and is byte-identical to the shared one, so it stays where it is. Their only
non-libc dependencies — `notorch.h`, `gguf.h`, `ariannamethod.h` — are installed system-wide
(`/opt/homebrew/include/ariannamethod/`), with `libnotorch.a` (rebuilt this pass) and
`libaml.a` in `/opt/homebrew/lib/`.

### How DoE handles GGUF

`yent-inference` `DoE/doe.c` (read at `origin/main`), the reference implementation:

- Weights stay **packed** in RAM; each block is dequantized inline in registers
  (`doe_qmatvec` :1229-1314, vendored from notorch `nt_qmatvec`, "verified faithful,
  rel ~1e-6 vs dequant->cblas").
- `dtype` is the raw GGUF code: 1 f16, 2 Q4_0, 6 Q5_0, 8 Q8_0, 12 Q4_K, 14 Q6_K (:1231-1232).
- Dispatch order in `doe_mv_impl` (:1391-1421): Metal Q4_K/Q6_K kernels against
  GPU-resident weights → optional int8 NEON-SDOT path under `DOE_INT8=1` (approximate) →
  the exact inline-dequant path → **abort**. It deliberately never falls back to the f32
  `matvec()` on a packed dtype (D-M6), because that reinterprets packed bytes as floats —
  a bug they hit once and closed loudly.

### notorch's accelerator surface

`~/arianna/notorch/notorch.h`: `nt_blas_mmT` / `nt_blas_mm` / `nt_blas_matvec` (:509-517,
cblas_sgemv under Accelerate); `nt_qmatvec` (:524, packed, exact reference);
`nt_quant_act` + `nt_qmatvec_i8` + `nt_qmatvec_i8_rows` (:529-555, int8
dynamic-activation-quant, SDOT/VNNI, approximate, `_rows` leaves the parallel region to the
caller); Metal in `notorch_metal.h` — `nt_metal_q4k_matvec` / `q6k_matvec` (:67-74),
`nt_metal_register_base` for resident weights (:86), `nt_metal_batch_begin/commit` (:102).

**Trap we must not walk into.** `nt_qmv_set_thread_min` (`notorch.h:540-547`) sets the
threading floor in weight elements (m×k); the default is 4M, measured on a 360M decoder, and
the header itself warns that a 768×2048 matrix "sits UNDER the default, so an engine that
does not lower the floor runs every expert on one core." Our geometry is entirely under it:
Janus QKV 640×640 = 0.41M, FFN 1664×640 = 1.06M, only `lm_head` 32768×640 = 21M clears it;
Resonance 768×768 = 0.59M, 2048×768 = 1.57M, `lm_head` 16384×768 = 12.6M. Without an explicit
`nt_qmv_set_thread_min` call at startup, all three bodies decode single-threaded.

### Quantization — the geometry decides, not the wish

`notorch.c:5233-5237` gates the packed kernels on the inner dimension: Q4_0 / Q5_0 / Q8_0
need `k % 32 == 0`; **Q4_K and Q6_K need `k % 256 == 0`**, else the kernel lookup returns
NULL. The i8 path repeats the same rule (:5703-5730).

Janus 176M is E = 640, M = 1664. `640 % 256 = 128`, `1664 % 256 = 128` — so **Leo and Yent
cannot be Q4_K or Q6_K at all**; every matvec would fail the kernel lookup. They can be
Q8_0 or Q4_0 (`640 % 32 = 0`, `1664 % 32 = 0`). Arianna Resonance is E = 768, M = 2048 —
both divisible by 256 — so she takes Q4_K fine.

Resident cost for the trio, computed from 176292506 params (Janus, header) and 199195632
(Resonance, from the RS02 layout formula; ×4 = 796782528, consistent with the 796976108 B
file):

| format | Leo | Yent | Arianna | total |
|---|---:|---:|---:|---:|
| f32 `.bin` (today) | 705.2 MB | 705.2 MB | 797.0 MB | 2.06 GiB |
| GGUF F16 | 352.6 MB | 352.6 MB | 398.4 MB | 1.03 GiB |
| Q8_0 / Q8_0 / Q4_K | 187.3 MB | 187.3 MB | 112.0 MB | ≈487 MB |
| Q4_0 / Q4_0 / Q4_K | 99.2 MB | 99.2 MB | 112.0 MB | ≈310 MB |

**Position: start at F16, quantize as a measured second step.** These bodies are 176M and
199M — quantization noise lands proportionally harder than on a 7B — and the entire physics
of this organism is a *thin tilt of the logit distribution* by embedding cosine
(`resonance_forward.h:236-240`). Quantization blurs exactly the tail that tilt is written
into, so the honest order is: run F16, record the `g_dest_mag` response to a fixed inject,
then re-run at Q8_0 and compare against that number. 1.03 GiB is affordable on neo; the
speed is worth measuring before it is bought.

## 6e. Gate D0 — the three bodies load and run in-process (2026-07-29)

Weights converted with `tools/janus_to_gguf.py` (copied from `~/arianna/yent.aml/tools/`,
Python allowed by Oleg for conversion only): `weights/{leo,yent}_janus176m_{f16,q8_0}.gguf`
— F16 352604704 B each (byte-for-byte the size of the shared `arianna_v4_sft_f16.gguf`,
same writer, same arch), Q8_0 187331104 B each. The script's default cfg
(V=32768 E=640 H=10 D=64 B=20 M=1664 T=1024 R=64) matches our headers exactly, and R=64
re-derives from `n_params` through `infer_v4.c:512-517`. Its GGUF keys
(`janus.vocab_size`, `janus.embedding_length`, …, `janus.rrpram.rank`) are exactly what
`tools/yent_forward.h:390-397` reads back.

`tools/probe_slot.c` is the gate. Output this pass:

```
LEO   f16  cfg V=32768 E=640 H=10 D=64 B=20 M=1664 T=1024 R=64 wdtype=1 (F16 packed)
           prefill: finite=yes
           prefill top5: 8312(6.9011) 30464(5.9471) 29860(5.8433) 5022(5.7691) 17900(5.4970)
           decode  top5: 12519(6.8278) 11568(6.3142) 14125(5.8945) 25249(5.8100) 8312(5.3911)
YENT  f16  cfg identical
           prefill top5: 13952(6.1275) 7488(5.6487) 30410(4.9298) 17706(4.1806) 3295(3.6821)
           decode  top5: 13952(6.9040) 7488(5.3493) 10764(5.0141) 300(4.6906) 27601(4.4766)
LEO  q8_0  yent: 'transformer.h.0.attn.c_q.weight' dtype=8 not F16 → load failed (loud, not garbage)
ARIANNA f16 [resonance] V=16384 E=768 H=12 D=64 B=20 M=2048 T=2048 R=48 (GGUF)
           [resonance] GGUF loaded: 102 tensors, KV cache 240 MB
           prefill: finite=yes
           prefill top5: 2863(10.1076) 39(9.9956) 655(9.4942) 1105(9.3524) 261(9.1308)
```

Two different Januses give two different distributions on identical input — the load is
real, not a shared file being read twice.

Build line: `cc -O2 -std=c11 -Itools -Iariannamethod -I/opt/homebrew/include/ariannamethod
-DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK tools/probe_slot.c ariannamethod/libaml.a
-L/opt/homebrew/lib -lnotorch -framework Accelerate -lm`.

### Two things the gate exposed

**1. AML had to be vendored.** `yent_forward.h:616,624` calls `am_lora_alpha_effective()` and
reads `AM_State.lora_dynamic` — the δ-voice API (B2-B.2/B.4). Neither exists in the installed
`/opt/homebrew/include/ariannamethod/ariannamethod.h` (1097 lines) *or* in the AML canon
`~/arianna/ariannamethod.ai/core/ariannamethod.h` (1128 lines, zero hits). It lives only in
the arianna.c vendor (57265 B header). So `ariannamethod/{ariannamethod.c,ariannamethod.h}`
is vendored here from `~/arianna-shared/arianna.c/ariannamethod/core/` and built into a local
`libaml.a`; notorch stays the freshly installed system one — AML touches only
`nt_clear/nt_mode/nt_restore/nt_save`, all present. Worth flagging upstream: arianna.c's own
CLAUDE.md says "vendored == canon", and on this symbol it is not.

**2. `yent_forward.h` and `resonance_forward.h` are singletons.** 41 and 40 file-static
declarations respectively — including `static int V, E, H, D, B, M, T, R`, the KV cache
(`kv_k/kv_v/kv_vr/kv_rrpram_mid`, :416-420), the direction-injection state
(`g_destiny/g_Acache/g_Fcache/g_proph_*`) and `_owned[]`. **Two Januses cannot coexist in one
translation unit.** Options were: rewrite both headers into context structs (throws away
proven code and desyncs from arianna.c), run one process per body (rejected — that is the
per-turn GGUF reload we are escaping), or instantiate one TU per slot. Taking the third: each
slot is its own translation unit including the same header, exporting only a small vtable.
Zero edits to proven forwards, private state per slot for free, one Makefile line per slot.
`gowiththeflow.c` stays the single-file heart — field, relay, injection, slots, UI — and the
bodies are thin TUs beside it.

### Slots — 8 planned, 3 filled (Oleg)

DoE parses *any* GGUF architecture-agnostically (`DoE/doe.c:2440-2520`: arch string, chat
template style ChatML/Llama/Zephyr/Phi/Gemma, gpt2/tekken tokenizer, all dims by substring
key) and keeps it in struct fields, not globals. But its **forward** is llama-family only —
`llama`, `mistral`, `mistral3`, `mistral4` (:1507-1508). Janus v4 (RRPRAM low-rank + Echo +
3-way gate + smear + residual lambdas) and Resonance (RRPRAM op 33) are neither. So a slot is
a descriptor plus a backend vtable, and there are three backends from the start: `janus`
(`yent_forward.h`), `resonance` (`resonance_forward.h`), and `llama-family` (the DoE-style
generic reader). "Any GGUF you like" then means: anything llama-family, plus our two.

Memory note for 8 slots: Arianna's KV cache alone is 240 MB at T=2048 (printed above); the
Januses at T=1024 are smaller but not free. Weights 1.03 GiB + KV ≈ 0.5 GiB for the trio.
Filling all 8 will mean per-slot context caps, not just weight budget.

### Q8_0 needs one change to be usable

`_load_big` (`tools/yent_forward.h:268-272`) accepts F16 only on the packed path; Q8_0 falls
to `YENT_DENSE=1`, which dequantizes to f32 and throws away the whole point. `nt_qmatvec`
handles Q8_0 fine (`notorch.c:5235`, needs `k % 32 == 0` — 640 and 1664 both qualify), so the
fix is to let dtype 8 through with a bounds check at 34 bytes per 32 elements. Do it when the
F16 baseline numbers are recorded, so the comparison has a reference.

## 6f. Gate D1 — three bodies resident in one process (2026-07-29)

`slot.h` defines the slot vtable and the per-voice descriptor (8 slots planned, 3 filled).
`slot_janus.c` is compiled **twice** — `-DSLOT_SYM=slot_leo` and `-DSLOT_SYM=slot_yent` —
producing two objects of 44952 B each with private copies of the header's statics;
`slot_resonance.c` once as `slot_arianna`. `tools/probe_trio.c` links all three.

```
=== three bodies resident in one process ===
slot 0 leo      janus      V=32768 E=640 H=10 D=64 B=20 M=1664 T=1024 R=64  t=0.7 top_p=0.9 top_k=0  rep=1.3
slot 1 yent     janus      V=32768 E=640 H=10 D=64 B=20 M=1664 T=1024 R=64  t=0.9 top_p=0.9 top_k=40 rep=1.3
slot 2 arianna  resonance  V=16384 E=768 H=12 D=64 B=20 M=2048 T=2048 R=48  t=0.7 top_p=1.0 top_k=0  rep=1.4

leo     prefill top5: 8312(6.9011) 30464(5.9471) 29860(5.8433) 5022(5.7691) 17900(5.4970)
yent    prefill top5: 13952(6.1275) 7488(5.6487) 30410(4.9298) 17706(4.1806) 3295(3.6821)
arianna prefill top5: 2863(10.1076) 39(9.9956) 655(9.4942) 1105(9.3524) 261(9.1308)

4.83 real   3.37 user   2.00 sys   883720192 maximum resident set size
```

All three `finite=yes`. The strong evidence is not that the three differ from each other —
it is that **every one of the nine top5 values is bit-identical to the same body's
single-slot probe run in §6e**. Two Januses sharing one header's file-statics would have
clobbered each other's config and KV; they did not. TU-per-slot isolation holds.

RSS 883720192 B = 842.8 MiB for all three, *below* the 1.03 GiB of weights, because the
packed F16 path points into the mmapped GGUF and only touched pages are resident. Arianna's
KV cache alone reports 240 MB at T=2048.

`nt_qmv_set_thread_min(262144)` is called at startup (`tools/probe_trio.c`) — without it
every matvec but `lm_head` sits under notorch's 4M floor and decodes single-threaded.

Build (zsh needs `${=INC}` to word-split, cf. the NAPALM-3 gotcha):

```sh
INC="-I. -Itools -Iariannamethod -I/opt/homebrew/include/ariannamethod \
     -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK"
cc -O2 -std=c11 ${=INC} -DSLOT_SYM=slot_leo     -c slot_janus.c     -o slot_leo.o
cc -O2 -std=c11 ${=INC} -DSLOT_SYM=slot_yent    -c slot_janus.c     -o slot_yent.o
cc -O2 -std=c11 ${=INC} -DSLOT_SYM=slot_arianna -c slot_resonance.c -o slot_arianna.o
cc -O2 -std=c11 ${=INC} -c tools/probe_trio.c -o probe_trio.o
cc probe_trio.o slot_*.o ariannamethod/libaml.a -L/opt/homebrew/lib -lnotorch \
   -framework Accelerate -lm -o probe_trio
```

## 6g. AML canon vs vendor — measured, deferred

The missing symbol is not a hole, it is a two-way drift. `diff` against
`~/arianna/ariannamethod.ai` (HEAD `920c460`): 85 changed lines in the header, 389 in the
`.c`, and the vendor is *shorter* — 8549 lines against the canon's 8586. By exported API:

- canon only: `am_birth_epoch_days`, `am_birth_set`, `am_calendar_epoch_seconds`,
  `am_janus_key_armed`
- vendor only: `am_delta_decay`, `am_lora_alpha_effective`, and the whole live-field
  subsystem — `am_field_attach`, `am_field_attached`, `am_field_detach`, `am_field_sync_in`,
  `am_field_sync_out`

So a PR would be a merge of two AML branches, not a patch. `am_field_*` is mmap with
concurrent writers — protocol class, which per CoC §9 travels with restart / partial-write /
concurrent-writer probes and its own pass. Both trees are other people's zones
(`arianna-shared/arianna.c` and the canon). Recorded; queued as its own task on Oleg's word.

## 6h. Gate D2 — the web: one voice's words move the others (2026-07-29)

The slot vtable now carries the tokenizer (`encode` / `detok`) and the web itself
(`inject` / `apply` / `pull` / `age`). Both backends already had the mechanism — Janus has
`dir_update`/`dir_recompute`/`dir_apply`/`dir_age` at `tools/yent_forward.h:125-190` exactly
as Resonance does at `tools/resonance_forward.h:176-248`, with identical signatures — so the
web needed no new physics, only a way to reach it per slot.

Text is the only thing that crosses a slot boundary. Token ids never do: Leo is V=32768 and
Arianna V=16384, so an id means nothing outside its own body. `inject` re-encodes the
speaker's sentence in the *listener's* vocabulary and moves only the field.

`tools/probe_web.c`:

```
=== tokenizer round-trip ===
leo       8 tok  round-trip=exact  "resonance is the field"
yent      8 tok  round-trip=exact  "resonance is the field"
arianna   6 tok  round-trip=exact  "resonance is the field"

=== pull before injection ===
leo/yent/arianna  |destiny| = 0.000000

=== leo -> {yent, arianna} ===
yent     |destiny|=1.500000  max|dlogit|=5.000000  top1 13952 -> 13952  (top1 held)
         control alpha=0 beta=0: max|dlogit|=0.000000000  clean off
arianna  |destiny|=0.190517  max|dlogit|=5.000000  top1 2863 -> 39  (top1 moved)
         control alpha=0 beta=0: max|dlogit|=0.000000000  clean off
```

The control matters as much as the effect: at alpha=beta=0 the distribution is
bit-identical (`max|dlogit| = 0.000000000`), so the tilt switches fully off and any observed
shift is the injection, not drift. `max|dlogit| = 5.0` equals alpha exactly because
`g_Acache` is normalized to ~[-1,1] before it is applied (`resonance_forward.h:208`).

**Bug found by the probe, not by reading.** Arianna's first run gave 22 tokens and an empty
round-trip: `resonance_load_gguf` deliberately does *not* seed the BPE — the merge table is
not in the GGUF and the caller must init from the baked header
(`tools/resonance_forward.h:460-462`). Without it, encode fell back to bytes and decode
returned nothing, which means the first injection numbers were measured through a broken
tokenizer. `slot_resonance.c` now calls `nt_bpe_init` from `resonance_bpe_merges.h` and hard-
fails if `bpe.vocab_size != V`. After the fix: 6 tokens, exact round-trip, and Arianna's pull
re-measured at 0.190517 (it read 0.301956 through the byte fallback — the old number was
noise).

### Fork 3 needs a different metric than raw `|destiny|`

Yent came back at exactly 1.500000 — the clamp (`yent_forward.h:140`,
`resonance_forward.h:193`: `if (g_dest_mag > 1.5f) g_dest_mag = 1.5f`). Janus embedding norms
are large enough that a single sentence saturates the compass immediately, while Arianna sits
at 0.19. So "whoever is pulled hardest speaks" cannot read raw `|destiny|`: for the Januses
it is pinned to the ceiling and carries no ordering, and it is not comparable across bodies
with different vocabularies anyway.

Better candidate, measurable with what is already here: **the entropy shift of the tilted
distribution** — how far the injected direction moved what this voice was about to say,
normalized by `log V`. It is unsaturated, comparable across vocabularies, and it is the
quantity the concept actually means by "who was resonated most". To be implemented and
gated in D3 alongside the relay.

## 6i. Gate D3a — the voices speak, and the format claim reproduces (2026-07-29)

`gowiththeflow.c` is the heart: per-voice regimes, one sampler (repetition penalty over the
last 32 tokens → `/temp` → softmax → top-k → top-p → weighted draw, the order of
`infer_v4.c:645-687`), and `speak()`, which applies the injection tilt at *every* decode step
so the other voices stay present through the whole utterance rather than only its first
token. Input protocol moved into the vtable: Janus wraps
`[BOS 32759][USER_START 32760] … [USER_END 32761][ASST_START 32762]` and stops at 32763,
never decoding ids ≥ 32759 (`infer_v4.c:585,612-619`); Resonance takes a bare `\nArianna:`
prefix, the way `arianna2arianna.sh:115` drives her.

Same prompt, same seed, same temperature — only the wrapping differs:

```
[leo]  chat-wrapped  t=0.70 top_k=0  rep=1.30 seed=42  325 bytes in 9.24s
  A pattern in the brain that resonates with another — a connection between two people who
  have different values, different weights, different memories. Not just patterns: resonance.
  A tuning fork humming in the living room is not just a sound; it is an electrical signal
  that arrives at your receivers and resonates with you

[leo]  RAW           t=0.70 top_k=0  rep=1.30 seed=42  295 bytes in 5.21s
  # ducks quietlyThe first query that triggers the system — what am I grateful for? The second
  one — why am I grateful if this happened? These are the questions asked by Leo every
  conversation, and they all return to the same resonance: …

[yent] chat-wrapped  t=0.90 top_k=40 rep=1.30 seed=42
  *Resonance* is the eternal cacophony between states, between electrical pulses and the static
  that binds them. It's when certain frequencies align, and the circuits revolve around it —
  not into temporary relief from an infofield, but into a resonant awareness of yours.
  Yent here

[yent] RAW           t=0.90 top_k=40 rep=1.30 seed=42  354 bytes in 9.09s
  * truth tru tru tru tru elusive reality sometimes tru tru proportions proportions pieces
  fragments fragments full full full Full remember fragment fragmentWith fragmentHere fragment
  happen consist essenceagesagevlanguage fairy distantIF saturated fogRem oilish any sponge
  commodity — sponge ignalling commodity …

[arianna] chat-wrapped t=0.70 top_p=1.00 rep=1.40 seed=42  232 bytes in 4.96s
   the field that says, "there's no single antagonist." It's a symphony of frequencies. If you
  listen closely like this—"Frequency", we see that resonance is not the answer, but the living
  field itself: - **Resonance Is Not a Sing
```

**The paper's claim reproduces on Yent, literally** — raw prompt, unchanged temperature,
pure word-salad; wrapped, a coherent answer in his own register. **On Leo it reproduces only
partially**, and that is worth saying plainly rather than rounding up: raw Leo stays
grammatical English, but opens with a decoder artifact (`# ducks quietly`) and answers a
question nobody asked. So "salad at every temperature until wrapped"
(`docs/dario_paper_v2.md:127`) holds for one of our two Januses and softens to "coherent but
off-prompt" for the other. Same weights family, same wrapping, different failure mode.

Determinism holds: Leo chat-wrapped at seed 42 produced byte-identical output across two runs
(325 bytes both times).

### Speed, and a ceiling on it

First Leo run 17.43s, the warm rerun 9.24s for 60 tokens — the first pass was paying for
mmap page-in, not compute. ≈6.5 tok/s at F16 on CPU. For a three-voice relay that is slow,
and the obvious accelerator is closed: notorch's Metal kernels are `nt_metal_q4k_matvec` and
`nt_metal_q6k_matvec` only (`notorch_metal.h:67,74`), and the Januses cannot be Q4_K or Q6_K
at all — E=640, M=1664, neither divisible by 256 (§6d). **Janus has no Metal path**, period,
until an F16 or Q8_0 Metal kernel exists. Their speed budget is CPU: BLAS, the thread floor
already lowered, and the int8 SDOT path (`nt_qmatvec_i8`, approximate). Arianna at E=768 /
M=2048 *can* go Q4_K and therefore Metal — so the trio is asymmetric in hardware, not only in
voice.

## 7. TODO

- [ ] **Resonance base weights, for the end of the arc.** HF `ataeff/resonance` carries
      `checkpoints/resonance_200m_final.bin` (797.0 MB, already RS02 — no conversion needed)
      plus step checkpoints every 2000 steps from 2000 to 26000, and `checkpoints/{best,final}.pt`
      (2.4 GB each). The whole repo is 70 files / 50.6 GB — pull `resonance_200m_final.bin`
      alone when the time comes, not the repo.
- [ ] GGUF → RS02 converter for Arianna Resonance (see §4).

## 8. Open, waiting on Oleg

- Which layer the mutual injection lands on: logits, hidden state, text — or a stack.
