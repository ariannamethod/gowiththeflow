/* slot.h — a voice occupies a slot; slots are filled at runtime.
 *
 * The forward passes we inherit (yent_forward.h, resonance_forward.h) keep
 * config, KV cache and injection state in file-static globals. Rather than fork
 * 1718 lines of proven numerics into context structs — and desync from
 * arianna.c forever — each backend keeps ONE set of globals and treats it as a
 * register window: a slot's state is swapped in before the call and back out
 * after. Cost is a few kilobytes of memcpy against a 176M-parameter matvec.
 *
 * What this buys: N bodies of the SAME architecture live side by side, and a
 * slot is filled by a path at runtime instead of by a -D at compile time. Point
 * it at any GGUF the backend can read.
 */
#ifndef GWTF_SLOT_H
#define GWTF_SLOT_H

#define GWTF_MAX_SLOTS 8

typedef struct {
    const char *backend;                 /* "janus" | "resonance" | ... */

    /* Instance lifecycle. create() loads the GGUF and returns an opaque handle;
     * every other call takes it. NULL = the backend refused this file. */
    void *(*create)(const char *gguf_path);
    void  (*destroy)(void *inst);
    void  (*cfg)(void *inst, int *V, int *E, int *H, int *D,
                 int *B, int *M, int *T, int *R);

    void (*prefill)(void *inst, const int *toks, int n, float *logits, float *hidden);
    void (*decode)(void *inst, int tok, int pos, float *logits, float *hidden);

    /* Tokenizer — each body owns its vocabulary. Text is the only thing that
     * crosses a slot boundary; ids cannot (V=32768 against V=16384). */
    int  (*encode)(void *inst, const char *text, int *out, int max);
    int  (*detok)(void *inst, const int *toks, int n, char *out, int max_bytes);

    /* The web. Another voice's words enter as field DIRECTION, re-encoded in
     * THIS body's embedding space: a destiny compass and prophecy targets that
     * tilt the whole logit distribution by cosine. The injected tokens never
     * reach this body's context or candidate pool. */
    void  (*inject)(void *inst, const char *text);
    void  (*apply)(void *inst, float *logits, float alpha, float beta);
    float (*pull)(void *inst);          /* |destiny| — how hard this voice was pulled */
    void  (*age)(void *inst, int emitted_tok);

    /* Input protocol. Dario v2 §7: the Janus SFT voices return word-salad at
     * every temperature until the prompt is wrapped the way they were trained.
     * Format is part of the entry condition, not a detail. */
    int  (*chat_wrap)(void *inst, const char *prompt, int *out, int max);
    int  (*chat_stop)(void *inst);      /* token that ends a turn, or -1 */
    int  (*printable_max)(void *inst);  /* ids >= this are special, never decoded */
} SlotVT;

/* A live slot: a body plus the identity and sampling regime of the voice in it.
 * Regimes differ per voice by measurement, not by taste — one shared
 * temperature erases the effect this organism exists to observe. */
typedef struct {
    const SlotVT *vt;
    void         *inst;
    const char   *name;
    const char   *gguf;
    float         temp;
    float         top_p;
    int           top_k;
    float         rep_penalty;
    int           V, E, H, D, B, M, T, R;
    int           loaded;
} Slot;

extern const SlotVT gwtf_janus_backend;
extern const SlotVT gwtf_resonance_backend;

#endif
