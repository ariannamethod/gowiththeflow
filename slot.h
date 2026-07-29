/* slot.h — one voice = one slot = one translation unit.
 *
 * yent_forward.h and resonance_forward.h keep their config, KV cache and
 * direction-injection state in file-static globals, so two Januses cannot live
 * in one TU. Instead of rewriting proven forwards, each slot compiles the same
 * header into its own object and exports nothing but this vtable. Private state
 * per slot comes for free; adding a slot is one Makefile line.
 */
#ifndef GWTF_SLOT_H
#define GWTF_SLOT_H

#define GWTF_MAX_SLOTS 8

typedef struct {
    const char *backend;                 /* "janus" | "resonance" | ... */
    int  (*load)(const char *gguf_path); /* 0 = ok */
    void (*cfg)(int *V, int *E, int *H, int *D, int *B, int *M, int *T, int *R);
    void (*prefill)(const int *toks, int n, float *logits, float *hidden);
    void (*decode)(int tok, int pos, float *logits, float *hidden);
    void (*release)(void);

    /* Tokenizer — each backend owns its own vocabulary. Text is the only thing
     * that crosses between slots; token ids never do (Leo is V=32768, Arianna
     * V=16384 — an id means nothing outside its own body). */
    int  (*encode)(const char *text, int *out, int max);
    int  (*detok)(const int *toks, int n, char *out, int max_bytes);

    /* The web. Another voice's words enter as field DIRECTION, encoded in THIS
     * body's embedding space: a destiny-EMA compass and prophecy targets that
     * tilt the whole logit distribution by cosine. The injected tokens never
     * enter this body's context or candidate pool. */
    void  (*inject)(const char *text);
    void  (*apply)(float *logits, float alpha, float beta);
    float (*pull)(void);          /* |destiny| — how hard this voice was pulled */
    void  (*age)(int emitted_tok);

    /* Input protocol. Dario v2 §7: the Janus SFT voices return word-salad at
     * every temperature until the prompt is wrapped the way they were trained.
     * Format is part of the entry condition, not a detail. */
    int  (*chat_wrap)(const char *prompt, int *out, int max);
    int  (*chat_stop)(void);      /* token that ends a turn, or -1 */
    int  (*printable_max)(void);  /* ids >= this are special, never decoded */
} SlotVT;

/* A live slot: a vtable plus the identity and sampling regime of the voice that
 * fills it. Regimes differ per voice by measurement, not by taste — a shared
 * temperature erases the effect this organism is built to observe. */
typedef struct {
    const SlotVT *vt;
    const char   *name;
    const char   *gguf;
    float         temp;
    float         top_p;
    int           top_k;
    float         rep_penalty;
    int           V, E, H, D, B, M, T, R;
    int           loaded;
} Slot;

#endif
