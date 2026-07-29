/* probe_slot — gate D0: prove a body loads from GGUF and runs a forward
 * in-process. One TU per body: yent_forward.h keeps its config and KV cache in
 * file-static state, so a second Janus needs a second translation unit, not a
 * second call. Build with -DPROBE_RESONANCE for the Resonance side. */
#include <stdio.h>
#include <stdlib.h>

#ifdef PROBE_RESONANCE
#include "resonance_forward.h"
#else
#include "yent_forward.h"
#endif

static void top5(const float *l, int n, const char *tag) {
    int idx[5] = {0};
    for (int k = 0; k < 5; k++) {
        float best = -1e30f; int bi = 0;
        for (int i = 0; i < n; i++) {
            int seen = 0;
            for (int j = 0; j < k; j++) if (idx[j] == i) seen = 1;
            if (!seen && l[i] > best) { best = l[i]; bi = i; }
        }
        idx[k] = bi;
    }
    printf("%s top5:", tag);
    for (int k = 0; k < 5; k++) printf(" %d(%.4f)", idx[k], l[idx[k]]);
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 2; }

    am_init();

#ifdef PROBE_RESONANCE
    ResonanceCtx ctx;
    if (resonance_load_gguf(&ctx, argv[1])) { fprintf(stderr, "load failed\n"); return 1; }
    Weights *w = &ctx.w;
    int wd = ctx.w.wdtype;
#else
    if (yent_read_cfg(argv[1])) { fprintf(stderr, "cfg read failed\n"); return 1; }
    Weights wbuf;
    if (yent_load_gguf(&wbuf, argv[1])) { fprintf(stderr, "load failed\n"); return 1; }
    Weights *w = &wbuf;
    int wd = wbuf.wdtype;
#endif

    printf("cfg V=%d E=%d H=%d D=%d B=%d M=%d T=%d R=%d wdtype=%d (%s)\n",
           V, E, H, D, B, M, T, R, wd,
           wd == GGUF_TYPE_F16 ? "F16 packed" : wd == GGUF_TYPE_F32 ? "F32 dense" : "other");

    kv_init(T);

    int toks[4] = { 1, 100, 200, 300 };
    for (int i = 0; i < 4; i++) if (toks[i] >= V) toks[i] = V - 1;

    float *logits = calloc((size_t)V, sizeof(float));
    float *hidden = calloc((size_t)E, sizeof(float));
    if (!logits || !hidden) { fprintf(stderr, "oom\n"); return 1; }

#ifdef PROBE_RESONANCE
    /* resonance_forward.h has no batched prefill — feed the prompt token by token */
    for (int i = 0; i < 4; i++) forward_token(w, toks[i], i, logits, hidden);
#else
    prefill_batch(w, toks, 4, logits, hidden);
#endif

    int finite = 1;
    for (int i = 0; i < V; i++)
        if (!(logits[i] == logits[i]) || logits[i] > 1e30f || logits[i] < -1e30f) { finite = 0; break; }
    printf("prefill: finite=%s\n", finite ? "yes" : "NO");
    top5(logits, V, "prefill");

    forward_token(w, toks[3], 4, logits, hidden);
    top5(logits, V, "decode ");

    free(logits); free(hidden);
    return finite ? 0 : 1;
}
