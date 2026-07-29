/* slot_janus.c — Janus v4 176M backend. Compiled once per Janus voice with a
 * different -DSLOT_SYM, so Leo and Yent get separate copies of the header's
 * file-static config, KV cache and injection state. */
#include "tools/yent_forward.h"
#include "janus_v4_bpe_merges.h"
#include "slot.h"

#ifndef SLOT_SYM
#error "compile with -DSLOT_SYM=<symbol> — one instance per voice"
#endif

static Weights g_w;
static nt_bpe  g_bpe;
static int     g_have = 0;

static int j_load(const char *path) {
    if (yent_read_cfg(path)) return 1;
    if (yent_load_gguf(&g_w, path)) return 1;
    kv_init(T);
    nt_bpe_init(&g_bpe, janus_v4_bpe_merges, JANUS_V4_BPE_MERGES);
    g_have = 1;
    return 0;
}

static void j_cfg(int *v, int *e, int *h, int *d, int *b, int *m, int *t, int *r) {
    *v = V; *e = E; *h = H; *d = D; *b = B; *m = M; *t = T; *r = R;
}

static void j_prefill(const int *toks, int n, float *logits, float *hidden) {
    if (!g_have) return;
    prefill_batch(&g_w, (int *)toks, n, logits, hidden);
}

static void j_decode(int tok, int pos, float *logits, float *hidden) {
    if (!g_have) return;
    forward_token(&g_w, tok, pos, logits, hidden);
}

static void j_release(void) {
    if (!g_have) return;
    yent_free(&g_w);
    g_have = 0;
}

static int j_encode(const char *text, int *out, int max) {
    return nt_bpe_encode(&g_bpe, text, (int)strlen(text), out, max);
}

static int j_detok(const int *toks, int n, char *out, int max_bytes) {
    return nt_bpe_decode(&g_bpe, toks, n, out, max_bytes);
}

/* Another voice's words → this body's destiny compass and prophecy ring. The
 * tokens are encoded in OUR vocabulary and used only to move the field; they
 * never reach the context (yent_forward.h:89, anti-fraud invariant). */
static void j_inject(const char *text) {
    if (!g_have || !text || !*text) return;
    int toks[512];
    int n = nt_bpe_encode(&g_bpe, text, (int)strlen(text), toks, 512);
    if (n <= 0) return;
    dir_update(g_w.wte, toks, n);
    dir_recompute(g_w.wte);
}

static void j_apply(float *logits, float alpha, float beta) {
    if (!g_have) return;
    dir_apply(logits, alpha, beta);
}

static float j_pull(void) { return g_dest_mag; }

static void j_age(int emitted) { if (g_have) dir_age(emitted); }

/* Janus SFT chat format (infer_v4.c:585, 612-619). Without this wrapping the
 * voice returns salad at every temperature — dario_paper_v2.md:127. */
#define TOK_BOS         32759
#define TOK_USER_START  32760
#define TOK_USER_END    32761
#define TOK_ASST_START  32762
#define TOK_ASST_END    32763

static int j_chat_wrap(const char *prompt, int *out, int max) {
    if (max < 8) return -1;
    int n = 0;
    out[n++] = TOK_BOS;
    out[n++] = TOK_USER_START;
    int body = nt_bpe_encode(&g_bpe, prompt, (int)strlen(prompt), out + n, max - n - 2);
    if (body < 0) return -1;
    n += body;
    out[n++] = TOK_USER_END;
    out[n++] = TOK_ASST_START;
    return n;
}

static int j_chat_stop(void)     { return TOK_ASST_END; }
static int j_printable_max(void) { return TOK_BOS; }

const SlotVT SLOT_SYM = {
    .backend = "janus",
    .load    = j_load,
    .cfg     = j_cfg,
    .prefill = j_prefill,
    .decode  = j_decode,
    .release = j_release,
    .encode  = j_encode,
    .detok   = j_detok,
    .inject  = j_inject,
    .apply   = j_apply,
    .pull    = j_pull,
    .age     = j_age,
    .chat_wrap     = j_chat_wrap,
    .chat_stop     = j_chat_stop,
    .printable_max = j_printable_max,
};
