/* gowiththeflow.c — the heart.
 *
 * A resonating chain, not a reasoning one: voices speak into each other and the
 * stream keeps moving after the human stops. Everything injects into everything —
 * a speaker's sentence enters every other slot as field DIRECTION (destiny
 * compass + prophecy targets), re-encoded in that listener's own vocabulary,
 * never as pasted tokens.
 *
 * Bodies live in slots (slot.h): one translation unit each, resident, packed
 * F16 out of the mapped GGUF.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "slot.h"
#include "notorch.h"
#include "ariannamethod.h"

extern const SlotVT slot_leo;
extern const SlotVT slot_yent;
extern const SlotVT slot_arianna;

/* Per-voice regimes. These are entry conditions, not preferences: a single
 * shared temperature erases the effect this organism exists to observe
 * (dario_paper_v2.md:127,145). Champions from cmd/internal/voices/voices.go:50-64
 * and scripts/arianna2arianna.sh:21-28 — treated as a starting hypothesis, since
 * the v2 sweep did not re-derive the exact cells. */
static Slot g_slots[GWTF_MAX_SLOTS] = {
    { .vt = &slot_leo,     .name = "leo",     .gguf = "weights/leo_janus176m_f16.gguf",
      .temp = 0.7f, .top_p = 1.0f, .top_k = 0,  .rep_penalty = 1.3f },
    { .vt = &slot_yent,    .name = "yent",    .gguf = "weights/yent_janus176m_f16.gguf",
      .temp = 0.9f, .top_p = 1.0f, .top_k = 40, .rep_penalty = 1.3f },
    { .vt = &slot_arianna, .name = "arianna", .gguf = "weights/arianna_resonance_v3_f16.gguf",
      .temp = 0.7f, .top_p = 1.0f, .top_k = 0,  .rep_penalty = 1.4f },
};
static int g_n_slots = 3;

/* ── sampling ──────────────────────────────────────────────────────────────
 * Order follows infer_v4.c:645-687 — repetition penalty over the last 32
 * emitted tokens, divide by temperature, softmax, top-k cut with renormalise,
 * then top-p nucleus, then a weighted draw. */

static void softmax_inplace(float *p, int n) {
    float mx = p[0];
    for (int i = 1; i < n; i++) if (p[i] > mx) mx = p[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { p[i] = expf(p[i] - mx); sum += p[i]; }
    if (sum > 0) for (int i = 0; i < n; i++) p[i] /= sum;
}

static int cmp_desc(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa < fb) - (fa > fb);
}

static int sample(float *logits, int V, const Slot *s, const int *hist, int hn) {
    int win = hn > 32 ? 32 : hn;
    for (int j = hn - win; j < hn; j++) {
        int t = hist[j];
        if (t >= 0 && t < V)
            logits[t] = logits[t] > 0 ? logits[t] / s->rep_penalty
                                      : logits[t] * s->rep_penalty;
    }
    for (int i = 0; i < V; i++) logits[i] /= s->temp;
    softmax_inplace(logits, V);

    if (s->top_k > 0 && s->top_k < V) {
        float *srt = malloc((size_t)V * sizeof(float));
        if (srt) {
            memcpy(srt, logits, (size_t)V * sizeof(float));
            qsort(srt, V, sizeof(float), cmp_desc);
            float thresh = srt[s->top_k - 1];
            free(srt);
            float z = 0;
            for (int i = 0; i < V; i++) { if (logits[i] < thresh) logits[i] = 0; z += logits[i]; }
            if (z > 0) for (int i = 0; i < V; i++) logits[i] /= z;
        }
    }

    if (s->top_p > 0.0f && s->top_p < 1.0f) {
        float *srt = malloc((size_t)V * sizeof(float));
        if (srt) {
            memcpy(srt, logits, (size_t)V * sizeof(float));
            qsort(srt, V, sizeof(float), cmp_desc);
            float cum = 0, cut = 0;
            for (int i = 0; i < V; i++) { cum += srt[i]; if (cum >= s->top_p) { cut = srt[i]; break; } }
            free(srt);
            float z = 0;
            for (int i = 0; i < V; i++) { if (logits[i] < cut) logits[i] = 0; z += logits[i]; }
            if (z > 0) for (int i = 0; i < V; i++) logits[i] /= z;
        }
    }

    float r = (float)rand() / (float)RAND_MAX, cum = 0;
    for (int i = 0; i < V; i++) { cum += logits[i]; if (cum >= r) return i; }
    return V - 1;
}

/* ── one turn ──────────────────────────────────────────────────────────────
 * inject_alpha/beta tilt every step of the generation, so the other voices are
 * present throughout the utterance rather than only in its first token.
 * alpha=0 leaves the distribution bit-identical (verified in probe_web). */
static int speak(Slot *s, const char *prompt, int max_tok,
                 float inject_alpha, float inject_beta,
                 int raw, char *out, int out_sz) {
    int V = s->V, E = s->E, T = s->T;
    int *ctx = malloc(sizeof(int) * (size_t)T);
    float *logits = calloc((size_t)V, sizeof(float));
    float *hidden = calloc((size_t)E, sizeof(float));
    if (!ctx || !logits || !hidden) return -1;

    int len = raw ? s->vt->encode(prompt, ctx, T)
                  : s->vt->chat_wrap(prompt, ctx, T);
    if (len <= 0) { free(ctx); free(logits); free(hidden); return -1; }
    if (len > T) len = T;

    s->vt->prefill(ctx, len, logits, hidden);

    int stop = s->vt->chat_stop();
    int pmax = s->vt->printable_max();
    int written = 0;
    out[0] = '\0';

    for (int step = 0; step < max_tok && len < T; step++) {
        s->vt->apply(logits, inject_alpha, inject_beta);
        int next = sample(logits, V, s, ctx, len);

        if (stop >= 0 && next == stop) break;

        if (next < pmax) {
            char piece[64];
            int nb = s->vt->detok(&next, 1, piece, (int)sizeof(piece) - 1);
            if (nb > 0) {
                piece[nb] = '\0';
                if (written + nb < out_sz - 1) {
                    memcpy(out + written, piece, (size_t)nb);
                    written += nb;
                    out[written] = '\0';
                }
            }
        }

        ctx[len++] = next;
        s->vt->age(next);
        s->vt->decode(next, len - 1, logits, hidden);
    }

    free(ctx); free(logits); free(hidden);
    return written;
}

static Slot *find_slot(const char *name) {
    for (int i = 0; i < g_n_slots; i++)
        if (strcmp(g_slots[i].name, name) == 0) return &g_slots[i];
    return NULL;
}

static int load_all(void) {
    for (int i = 0; i < g_n_slots; i++) {
        if (g_slots[i].vt->load(g_slots[i].gguf)) {
            fprintf(stderr, "slot %s: load failed (%s)\n", g_slots[i].name, g_slots[i].gguf);
            return 1;
        }
        g_slots[i].vt->cfg(&g_slots[i].V, &g_slots[i].E, &g_slots[i].H, &g_slots[i].D,
                           &g_slots[i].B, &g_slots[i].M, &g_slots[i].T, &g_slots[i].R);
        g_slots[i].loaded = 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    unsigned seed = 1;
    int max_tok = 60, raw = 0;
    const char *mode = NULL, *who = NULL, *prompt = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--speak") && i + 2 < argc) { mode = "speak"; who = argv[++i]; prompt = argv[++i]; }
        else if (!strcmp(argv[i], "--raw"))                raw = 1;
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc)     max_tok = atoi(argv[++i]);
    }
    if (!mode) {
        fprintf(stderr, "usage: %s --speak <leo|yent|arianna> \"prompt\" [--raw] [--seed N] [-n TOK]\n", argv[0]);
        return 2;
    }

    am_init();
    /* Every matvec here but lm_head is under notorch's 4M threading floor
     * (notorch.h:540-547); without this the bodies decode on one core. */
    nt_qmv_set_thread_min(262144);

    if (load_all()) return 1;
    srand(seed);

    Slot *s = find_slot(who);
    if (!s) { fprintf(stderr, "no slot named '%s'\n", who); return 2; }

    static char out[1 << 16];
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int n = speak(s, prompt, max_tok, 0.0f, 0.0f, raw, out, sizeof(out));
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double el = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    printf("\n[%s] %s  t=%.2f top_k=%d top_p=%.2f rep=%.2f seed=%u  %d bytes in %.2fs\n",
           s->name, raw ? "RAW (no chat wrap)" : "chat-wrapped",
           s->temp, s->top_k, s->top_p, s->rep_penalty, seed, n, el);
    printf("%s\n", n > 0 ? out : "(nothing)");

    for (int i = 0; i < g_n_slots; i++) g_slots[i].vt->release();
    return 0;
}
