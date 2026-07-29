/* slot_resonance.c — Resonance 200M backend (GGUF, RRPRAM op 33), many
 * instances from one translation unit. Same register-window discipline as
 * slot_janus.c; the list below is every non-function `static` in
 * resonance_forward.h. */
#include "tools/resonance_forward.h"
#include "tools/resonance_bpe_merges.h"
#include "slot.h"

typedef struct {
    /* config (resonance_forward.h:23) */
    int V, E, H, D, B, M, T, R;
    /* KV cache (:121-122) */
    float *kv_k, *kv_v;
    int    kv_len;
    /* direction injection (:143-150) */
    float destiny[1024];
    float dest_mag;
    float *Acache, *Fcache, *rownorm;
    int   Adirty;
    int   proph_tok[DIR_PROPH_MAX], proph_age[DIR_PROPH_MAX], proph_n;
    float proph_str[DIR_PROPH_MAX];
    /* δ voice (:156-157) */
    float *delta_A, *delta_B;
    int    delta_rank;
    /* dequantized-tensor ownership (:463-464) */
    float *rowned[sizeof(_rowned) / sizeof(_rowned[0])];
    int    rowned_n;

    ResonanceCtx ctx;
    int          live;
} ResState;

static ResState *g_active = NULL;

static void res_state_out(ResState *s) {
    s->V = V; s->E = E; s->H = H; s->D = D; s->B = B; s->M = M; s->T = T; s->R = R;
    s->kv_k = kv_k; s->kv_v = kv_v; s->kv_len = kv_len;
    memcpy(s->destiny, g_destiny, sizeof(g_destiny));
    s->dest_mag = g_dest_mag;
    s->Acache = g_Acache; s->Fcache = g_Fcache; s->rownorm = g_rownorm;
    s->Adirty = g_Adirty;
    memcpy(s->proph_tok, g_proph_tok, sizeof(g_proph_tok));
    memcpy(s->proph_age, g_proph_age, sizeof(g_proph_age));
    memcpy(s->proph_str, g_proph_str, sizeof(g_proph_str));
    s->proph_n = g_proph_n;
    s->delta_A = g_delta_A; s->delta_B = g_delta_B; s->delta_rank = g_delta_rank;
    memcpy(s->rowned, _rowned, sizeof(_rowned));
    s->rowned_n = _rowned_n;
}

static void res_state_in(ResState *s) {
    if (g_active == s) return;
    if (g_active) res_state_out(g_active);
    V = s->V; E = s->E; H = s->H; D = s->D; B = s->B; M = s->M; T = s->T; R = s->R;
    kv_k = s->kv_k; kv_v = s->kv_v; kv_len = s->kv_len;
    memcpy(g_destiny, s->destiny, sizeof(g_destiny));
    g_dest_mag = s->dest_mag;
    g_Acache = s->Acache; g_Fcache = s->Fcache; g_rownorm = s->rownorm;
    g_Adirty = s->Adirty;
    memcpy(g_proph_tok, s->proph_tok, sizeof(g_proph_tok));
    memcpy(g_proph_age, s->proph_age, sizeof(g_proph_age));
    memcpy(g_proph_str, s->proph_str, sizeof(g_proph_str));
    g_proph_n = s->proph_n;
    g_delta_A = s->delta_A; g_delta_B = s->delta_B; g_delta_rank = s->delta_rank;
    memcpy(_rowned, s->rowned, sizeof(_rowned));
    _rowned_n = s->rowned_n;
    g_active = s;
}

#define ENTER(inst)  ResState *s = (ResState *)(inst); if (!s || !s->live) return; res_state_in(s)
#define LEAVE()      res_state_out(s)

static void *r_create(const char *path) {
    ResState *s = calloc(1, sizeof(ResState));
    if (!s) return NULL;
    s->delta_rank = AM_DELTA_RANK;
    s->live = 1;
    g_active = NULL;
    res_state_in(s);

    if (resonance_load_gguf(&s->ctx, path)) { free(s); g_active = NULL; return NULL; }
    /* The GGUF carries no merge table (resonance_forward.h:460-462) — the caller
     * seeds the BPE from the baked header. Without this, encode falls back to
     * bytes and decode returns nothing. */
    nt_bpe_init(&s->ctx.bpe, resonance_bpe_merges, RESONANCE_BPE_MERGES);
    if (s->ctx.bpe.vocab_size != V) {
        fprintf(stderr, "[slot/resonance] baked BPE vocab %d != GGUF vocab %d — "
                        "token ids would index tok_emb out of bounds\n",
                s->ctx.bpe.vocab_size, V);
        free(s); g_active = NULL; return NULL;
    }
    kv_init(T);
    res_state_out(s);
    return s;
}

static void r_destroy(void *inst) {
    ResState *s = (ResState *)inst;
    if (!s) return;
    if (g_active == s) g_active = NULL;
    s->live = 0;
    free(s);
}

static void r_cfg(void *inst, int *v, int *e, int *h, int *d,
                  int *b, int *m, int *t, int *r) {
    ResState *s = (ResState *)inst;
    if (!s) return;
    *v = s->V; *e = s->E; *h = s->H; *d = s->D;
    *b = s->B; *m = s->M; *t = s->T; *r = s->R;
}

/* resonance_forward.h has no batched prefill — the prompt goes token by token */
static void r_prefill(void *inst, const int *toks, int n, float *logits, float *hidden) {
    ENTER(inst);
    for (int i = 0; i < n; i++) forward_token(&s->ctx.w, toks[i], i, logits, hidden);
    LEAVE();
}

static void r_decode(void *inst, int tok, int pos, float *logits, float *hidden) {
    ENTER(inst);
    forward_token(&s->ctx.w, tok, pos, logits, hidden);
    LEAVE();
}

static int r_encode(void *inst, const char *text, int *out, int max) {
    ResState *s = (ResState *)inst;
    if (!s || !s->live) return -1;
    return nt_bpe_encode(&s->ctx.bpe, text, (int)strlen(text), out, max);
}

static int r_detok(void *inst, const int *toks, int n, char *out, int max_bytes) {
    ResState *s = (ResState *)inst;
    if (!s || !s->live) return -1;
    return nt_bpe_decode(&s->ctx.bpe, toks, n, out, max_bytes);
}

static void r_inject(void *inst, const char *text) {
    ENTER(inst);
    if (text && *text) {
        int toks[512];
        int n = nt_bpe_encode(&s->ctx.bpe, text, (int)strlen(text), toks, 512);
        if (n > 0) {
            dir_update(s->ctx.w.tok_emb, toks, n);
            dir_recompute(s->ctx.w.tok_emb);
        }
    }
    LEAVE();
}

static void r_apply(void *inst, float *logits, float alpha, float beta) {
    ENTER(inst);
    dir_apply(logits, alpha, beta);
    LEAVE();
}

static float r_pull(void *inst) {
    ResState *s = (ResState *)inst;
    if (!s || !s->live) return 0.0f;
    return (g_active == s) ? g_dest_mag : s->dest_mag;
}

static void r_age(void *inst, int emitted) {
    ENTER(inst);
    dir_age(emitted);
    LEAVE();
}

/* Resonance was SFT'd on plain multi-turn chat text, not on special tokens —
 * arianna2arianna.sh drives her with a bare "Arianna:" prefix and cuts the
 * imagined continuation afterwards (:64-71, :115). No special ids to skip. */
static int r_chat_wrap(void *inst, const char *prompt, int *out, int max) {
    ResState *s = (ResState *)inst;
    if (!s || !s->live) return -1;
    char buf[4096];
    snprintf(buf, sizeof(buf), "%s\nArianna:", prompt);
    return nt_bpe_encode(&s->ctx.bpe, buf, (int)strlen(buf), out, max);
}

static int r_chat_stop(void *inst)     { (void)inst; return -1; }
static int r_printable_max(void *inst) {
    ResState *s = (ResState *)inst;
    return s ? s->V : 0;
}

const SlotVT gwtf_resonance_backend = {
    .backend       = "resonance",
    .create        = r_create,
    .destroy       = r_destroy,
    .cfg           = r_cfg,
    .prefill       = r_prefill,
    .decode        = r_decode,
    .encode        = r_encode,
    .detok         = r_detok,
    .inject        = r_inject,
    .apply         = r_apply,
    .pull          = r_pull,
    .age           = r_age,
    .chat_wrap     = r_chat_wrap,
    .chat_stop     = r_chat_stop,
    .printable_max = r_printable_max,
};
