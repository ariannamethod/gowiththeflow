/* slot_janus.c — Janus v4 176M backend, many instances from one translation
 * unit.
 *
 * yent_forward.h holds its whole world in file-static globals. Instead of
 * rewriting it, an instance owns a JanusState and we swap that state into the
 * globals around every call. The list below is mechanical and must stay
 * complete: it is every non-function `static` in yent_forward.h. Miss one and
 * two voices silently share it.
 */
#include "tools/yent_forward.h"
#include "janus_v4_bpe_merges.h"
#include "slot.h"

typedef struct {
    /* config */
    int V, E, H, D, B, M, T, R;
    /* δ voice (yent_forward.h:29-30) */
    float *delta_A, *delta_B;
    int    delta_rank;
    /* direction injection (:91-96) */
    float destiny[1024];
    float dest_mag;
    float *Acache, *Fcache, *rownorm;
    int   Adirty;
    int   proph_tok[DIR_PROPH_MAX], proph_age[DIR_PROPH_MAX], proph_n;
    float proph_str[DIR_PROPH_MAX];
    /* dequantized-tensor ownership (:238-239) */
    _OwnedTensor owned[sizeof(_owned) / sizeof(_owned[0])];
    int          owned_n;
    /* KV cache (:416-420) */
    float *kv_k, *kv_v, *kv_vr, *kv_rrpram_mid;
    int    kv_len;

    /* per-instance, not part of the header's globals */
    Weights w;
    nt_bpe  bpe;
    int     live;
} JanusState;

static JanusState *g_active = NULL;

static void janus_state_out(JanusState *s) {
    s->V = V; s->E = E; s->H = H; s->D = D; s->B = B; s->M = M; s->T = T; s->R = R;
    s->delta_A = g_delta_A; s->delta_B = g_delta_B; s->delta_rank = g_delta_rank;
    memcpy(s->destiny, g_destiny, sizeof(g_destiny));
    s->dest_mag = g_dest_mag;
    s->Acache = g_Acache; s->Fcache = g_Fcache; s->rownorm = g_rownorm;
    s->Adirty = g_Adirty;
    memcpy(s->proph_tok, g_proph_tok, sizeof(g_proph_tok));
    memcpy(s->proph_age, g_proph_age, sizeof(g_proph_age));
    memcpy(s->proph_str, g_proph_str, sizeof(g_proph_str));
    s->proph_n = g_proph_n;
    memcpy(s->owned, _owned, sizeof(_owned));
    s->owned_n = _owned_n;
    s->kv_k = kv_k; s->kv_v = kv_v; s->kv_vr = kv_vr; s->kv_rrpram_mid = kv_rrpram_mid;
    s->kv_len = kv_len;
}

static void janus_state_in(JanusState *s) {
    if (g_active == s) return;
    if (g_active) janus_state_out(g_active);
    V = s->V; E = s->E; H = s->H; D = s->D; B = s->B; M = s->M; T = s->T; R = s->R;
    g_delta_A = s->delta_A; g_delta_B = s->delta_B; g_delta_rank = s->delta_rank;
    memcpy(g_destiny, s->destiny, sizeof(g_destiny));
    g_dest_mag = s->dest_mag;
    g_Acache = s->Acache; g_Fcache = s->Fcache; g_rownorm = s->rownorm;
    g_Adirty = s->Adirty;
    memcpy(g_proph_tok, s->proph_tok, sizeof(g_proph_tok));
    memcpy(g_proph_age, s->proph_age, sizeof(g_proph_age));
    memcpy(g_proph_str, s->proph_str, sizeof(g_proph_str));
    g_proph_n = s->proph_n;
    memcpy(_owned, s->owned, sizeof(_owned));
    _owned_n = s->owned_n;
    kv_k = s->kv_k; kv_v = s->kv_v; kv_vr = s->kv_vr; kv_rrpram_mid = s->kv_rrpram_mid;
    kv_len = s->kv_len;
    g_active = s;
}

/* Enter/leave around every entry point. Leaving writes the globals back, so a
 * call that mutated the field (dir_update, kv writes) is not lost. */
#define ENTER(inst)  JanusState *s = (JanusState *)(inst); if (!s || !s->live) return; janus_state_in(s)
#define ENTER_R(inst, r)  JanusState *s = (JanusState *)(inst); if (!s || !s->live) return (r); janus_state_in(s)
#define LEAVE()      janus_state_out(s)

static void *j_create(const char *path) {
    JanusState *s = calloc(1, sizeof(JanusState));
    if (!s) return NULL;

    /* Load into a clean window: the header's globals are shared scratch. */
    JanusState blank = {0};
    blank.delta_rank = AM_DELTA_RANK;
    blank.live = 1;
    *s = blank;
    g_active = NULL;
    janus_state_in(s);

    if (yent_read_cfg(path)) { free(s); g_active = NULL; return NULL; }
    if (yent_load_gguf(&s->w, path)) { free(s); g_active = NULL; return NULL; }
    kv_init(T);
    nt_bpe_init(&s->bpe, janus_v4_bpe_merges, JANUS_V4_BPE_MERGES);
    janus_state_out(s);
    return s;
}

static void j_destroy(void *inst) {
    JanusState *s = (JanusState *)inst;
    if (!s) return;
    janus_state_in(s);
    yent_free(&s->w);
    janus_state_out(s);
    if (g_active == s) g_active = NULL;
    s->live = 0;
    free(s);
}

static void j_cfg(void *inst, int *v, int *e, int *h, int *d,
                  int *b, int *m, int *t, int *r) {
    JanusState *s = (JanusState *)inst;
    if (!s) return;
    *v = s->V; *e = s->E; *h = s->H; *d = s->D;
    *b = s->B; *m = s->M; *t = s->T; *r = s->R;
}

static void j_prefill(void *inst, const int *toks, int n, float *logits, float *hidden) {
    ENTER(inst);
    prefill_batch(&s->w, (int *)toks, n, logits, hidden);
    LEAVE();
}

static void j_decode(void *inst, int tok, int pos, float *logits, float *hidden) {
    ENTER(inst);
    forward_token(&s->w, tok, pos, logits, hidden);
    LEAVE();
}

static int j_encode(void *inst, const char *text, int *out, int max) {
    JanusState *s = (JanusState *)inst;
    if (!s || !s->live) return -1;
    return nt_bpe_encode(&s->bpe, text, (int)strlen(text), out, max);
}

static int j_detok(void *inst, const int *toks, int n, char *out, int max_bytes) {
    JanusState *s = (JanusState *)inst;
    if (!s || !s->live) return -1;
    return nt_bpe_decode(&s->bpe, toks, n, out, max_bytes);
}

/* Another voice's words → this body's destiny compass and prophecy ring. The
 * tokens are encoded in OUR vocabulary and only move the field; they never
 * reach the context (yent_forward.h:89, anti-fraud invariant). */
static void j_inject(void *inst, const char *text) {
    ENTER(inst);
    if (text && *text) {
        int toks[512];
        int n = nt_bpe_encode(&s->bpe, text, (int)strlen(text), toks, 512);
        if (n > 0) {
            dir_update(s->w.wte, toks, n);
            dir_recompute(s->w.wte);
        }
    }
    LEAVE();
}

static void j_apply(void *inst, float *logits, float alpha, float beta) {
    ENTER(inst);
    dir_apply(logits, alpha, beta);
    LEAVE();
}

static float j_pull(void *inst) {
    JanusState *s = (JanusState *)inst;
    if (!s || !s->live) return 0.0f;
    return (g_active == s) ? g_dest_mag : s->dest_mag;
}

static void j_age(void *inst, int emitted) {
    ENTER(inst);
    dir_age(emitted);
    LEAVE();
}

/* Janus SFT chat format (infer_v4.c:585, 612-619). Without this wrapping the
 * voice returns salad at every temperature — dario_paper_v2.md:127. */
#define TOK_BOS         32759
#define TOK_USER_START  32760
#define TOK_USER_END    32761
#define TOK_ASST_START  32762
#define TOK_ASST_END    32763

static int j_chat_wrap(void *inst, const char *prompt, int *out, int max) {
    JanusState *s = (JanusState *)inst;
    if (!s || !s->live || max < 8) return -1;
    int n = 0;
    out[n++] = TOK_BOS;
    out[n++] = TOK_USER_START;
    int body = nt_bpe_encode(&s->bpe, prompt, (int)strlen(prompt), out + n, max - n - 2);
    if (body < 0) return -1;
    n += body;
    out[n++] = TOK_USER_END;
    out[n++] = TOK_ASST_START;
    return n;
}

static int j_chat_stop(void *inst)     { (void)inst; return TOK_ASST_END; }
static int j_printable_max(void *inst) { (void)inst; return TOK_BOS; }

const SlotVT gwtf_janus_backend = {
    .backend       = "janus",
    .create        = j_create,
    .destroy       = j_destroy,
    .cfg           = j_cfg,
    .prefill       = j_prefill,
    .decode        = j_decode,
    .encode        = j_encode,
    .detok         = j_detok,
    .inject        = j_inject,
    .apply         = j_apply,
    .pull          = j_pull,
    .age           = j_age,
    .chat_wrap     = j_chat_wrap,
    .chat_stop     = j_chat_stop,
    .printable_max = j_printable_max,
};
