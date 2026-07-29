/* slot_resonance.c — Resonance 200M backend (GGUF, RRPRAM op 33). Same
 * one-TU-per-voice rule as slot_janus.c; the header carries its own BPE inside
 * ResonanceCtx, and there is no batched prefill — the prompt is fed token by
 * token. */
#include "tools/resonance_forward.h"
#include "tools/resonance_bpe_merges.h"
#include "slot.h"

#ifndef SLOT_SYM
#error "compile with -DSLOT_SYM=<symbol> — one instance per voice"
#endif

static ResonanceCtx g_ctx;
static int          g_have = 0;

static int r_load(const char *path) {
    if (resonance_load_gguf(&g_ctx, path)) return 1;
    /* The GGUF carries no merge table (resonance_forward.h:460-462) — the caller
     * seeds the BPE from the baked header. Without this, encode falls back to
     * bytes and decode returns nothing. */
    nt_bpe_init(&g_ctx.bpe, resonance_bpe_merges, RESONANCE_BPE_MERGES);
    if (g_ctx.bpe.vocab_size != V) {
        fprintf(stderr, "[slot/resonance] baked BPE vocab %d != GGUF vocab %d — "
                        "token ids would index tok_emb out of bounds\n",
                g_ctx.bpe.vocab_size, V);
        return 1;
    }
    kv_init(T);
    g_have = 1;
    return 0;
}

static void r_cfg(int *v, int *e, int *h, int *d, int *b, int *m, int *t, int *r) {
    *v = V; *e = E; *h = H; *d = D; *b = B; *m = M; *t = T; *r = R;
}

static void r_prefill(const int *toks, int n, float *logits, float *hidden) {
    if (!g_have) return;
    for (int i = 0; i < n; i++) forward_token(&g_ctx.w, toks[i], i, logits, hidden);
}

static void r_decode(int tok, int pos, float *logits, float *hidden) {
    if (!g_have) return;
    forward_token(&g_ctx.w, tok, pos, logits, hidden);
}

static void r_release(void) {
    g_have = 0;
}

static int r_encode(const char *text, int *out, int max) {
    return nt_bpe_encode(&g_ctx.bpe, text, (int)strlen(text), out, max);
}

static int r_detok(const int *toks, int n, char *out, int max_bytes) {
    return nt_bpe_decode(&g_ctx.bpe, toks, n, out, max_bytes);
}

static void r_inject(const char *text) {
    if (!g_have || !text || !*text) return;
    int toks[512];
    int n = nt_bpe_encode(&g_ctx.bpe, text, (int)strlen(text), toks, 512);
    if (n <= 0) return;
    dir_update(g_ctx.w.tok_emb, toks, n);
    dir_recompute(g_ctx.w.tok_emb);
}

static void r_apply(float *logits, float alpha, float beta) {
    if (!g_have) return;
    dir_apply(logits, alpha, beta);
}

static float r_pull(void) { return g_dest_mag; }

static void r_age(int emitted) { if (g_have) dir_age(emitted); }

/* Resonance was SFT'd on plain multi-turn chat text, not on special tokens —
 * arianna2arianna.sh drives her with a bare "Arianna:" prefix and cuts the
 * imagined continuation afterwards (:64-71, :115). No special ids to skip. */
static int r_chat_wrap(const char *prompt, int *out, int max) {
    char buf[4096];
    snprintf(buf, sizeof(buf), "%s\nArianna:", prompt);
    return nt_bpe_encode(&g_ctx.bpe, buf, (int)strlen(buf), out, max);
}

static int r_chat_stop(void)     { return -1; }
static int r_printable_max(void) { return V; }

const SlotVT SLOT_SYM = {
    .backend = "resonance",
    .load    = r_load,
    .cfg     = r_cfg,
    .prefill = r_prefill,
    .decode  = r_decode,
    .release = r_release,
    .encode  = r_encode,
    .detok   = r_detok,
    .inject  = r_inject,
    .apply   = r_apply,
    .pull    = r_pull,
    .age     = r_age,
    .chat_wrap     = r_chat_wrap,
    .chat_stop     = r_chat_stop,
    .printable_max = r_printable_max,
};
