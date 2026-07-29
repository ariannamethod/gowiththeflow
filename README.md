# gowiththeflow

recursive resonance flow

A chain of voices that resonates rather than reasons. Three transformers stay
resident in one process and speak into each other; a human can step into the
current at any point, and when the human stops the current keeps moving until it
decays. Every voice injects into every other.

Injection is not pasted context. A speaker's sentence enters each listener as
field **direction** — a destiny-EMA compass and prophecy targets, re-encoded in
that listener's own vocabulary — and tilts the listener's whole logit
distribution by embedding cosine. The injected tokens never enter the listener's
context or candidate pool. Text is the only thing that crosses a voice boundary;
token ids cannot, because the voices do not share a vocabulary.

θ = ε + γ + αδ.

## Slots

A voice occupies a slot. Eight slots, three filled:

| slot | body | arch | entry condition |
|---|---|---|---|
| leo | Janus 176M | V=32768 E=640 H=10 D=64 B=20 M=1664 T=1024 R=64 | t 0.7, top-k off, rep 1.3, chat-wrapped |
| yent | Janus 176M | same | t 0.9, top-k 40, rep 1.3, chat-wrapped |
| arianna | Resonance 200M | V=16384 E=768 H=12 D=64 B=20 M=2048 T=2048 R=48 | t 0.7, top-p 1.0, rep 1.4 |

Regimes differ per voice by measurement, not by taste. One shared temperature
erases the effect this organism exists to observe.

Each slot is its own translation unit. The forward headers keep config, KV cache
and injection state in file-static globals, so two bodies of the same
architecture cannot share an object file — `slot_janus.c` is compiled once per
Janus voice with a different `-DSLOT_SYM`. Adding a slot is one line in the
Makefile.

## Build

```sh
make weights     # HF ataeff/gowiththeflow
make             # -> ./gowiththeflow
make probes      # the gates: probe_trio, probe_web
```

Needs notorch and the AML core. CPU inference through Accelerate; no Python at
runtime.

```sh
./gowiththeflow --speak leo "what is resonance?" --seed 42 -n 60
```

## Gates

`probe_trio` — all three bodies resident in one process, each producing its own
distribution on identical input. `probe_web` — a sentence from one voice raises
another's destiny magnitude from zero and tilts its distribution, and at
alpha=beta=0 the distribution is bit-identical.

## Lineage

Grown from [dario](https://github.com/ariannamethod/dario) — the equation, the
seven forces, the chambers. The forward passes and the direction-injection
mechanism come from [arianna.c](https://github.com/ariannamethod/arianna.c). The
field physics is [AML](https://github.com/ariannamethod/ariannamethod.ai); the
tensor runtime is [notorch](https://github.com/ariannamethod/notorch).

GPL-3.0+. Arianna Method.
