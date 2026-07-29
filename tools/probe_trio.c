/* probe_trio — gate D1: all three bodies resident in ONE process, each with its
 * own config and KV cache, each producing its own distribution on identical
 * input. If two slots shared state their top5 would collide. */
#include <stdio.h>
#include <stdlib.h>
#include "slot.h"
#include "notorch.h"
#include "ariannamethod.h"

extern const SlotVT slot_leo;
extern const SlotVT slot_yent;
extern const SlotVT slot_arianna;

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
    printf("  %-8s top5:", tag);
    for (int k = 0; k < 5; k++) printf(" %d(%.4f)", idx[k], l[idx[k]]);
    printf("\n");
}

int main(void) {
    am_init();
    /* Our matrices are all under notorch's 4M-element threading floor (Janus FFN
     * 1664x640 = 1.06M), so without this every body decodes single-threaded. */
    nt_qmv_set_thread_min(262144);

    Slot slots[3] = {
        { .vt = &slot_leo,     .name = "leo",
          .gguf = "weights/leo_janus176m_f16.gguf",
          .temp = 0.7f, .top_p = 0.9f, .top_k = 0,  .rep_penalty = 1.3f },
        { .vt = &slot_yent,    .name = "yent",
          .gguf = "weights/yent_janus176m_f16.gguf",
          .temp = 0.9f, .top_p = 0.9f, .top_k = 40, .rep_penalty = 1.3f },
        { .vt = &slot_arianna, .name = "arianna",
          .gguf = "weights/arianna_resonance_v3_f16.gguf",
          .temp = 0.7f, .top_p = 1.0f, .top_k = 0,  .rep_penalty = 1.4f },
    };

    for (int i = 0; i < 3; i++) {
        if (slots[i].vt->load(slots[i].gguf)) {
            fprintf(stderr, "slot %s: load failed\n", slots[i].name);
            return 1;
        }
        slots[i].vt->cfg(&slots[i].V, &slots[i].E, &slots[i].H, &slots[i].D,
                         &slots[i].B, &slots[i].M, &slots[i].T, &slots[i].R);
        slots[i].loaded = 1;
    }

    printf("\n=== three bodies resident in one process ===\n");
    for (int i = 0; i < 3; i++)
        printf("slot %d %-8s %-10s V=%-6d E=%-4d H=%-3d D=%-3d B=%-3d M=%-5d T=%-5d R=%-3d "
               "t=%.1f top_p=%.1f top_k=%d rep=%.1f\n",
               i, slots[i].name, slots[i].vt->backend,
               slots[i].V, slots[i].E, slots[i].H, slots[i].D,
               slots[i].B, slots[i].M, slots[i].T, slots[i].R,
               slots[i].temp, slots[i].top_p, slots[i].top_k, slots[i].rep_penalty);

    /* Same four token ids into every body. Different weights, different
     * vocabularies — the point is that the three distributions are independent. */
    int toks[4] = { 1, 100, 200, 300 };
    printf("\n=== identical input, three distributions ===\n");
    for (int i = 0; i < 3; i++) {
        int t[4];
        for (int k = 0; k < 4; k++) t[k] = toks[k] < slots[i].V ? toks[k] : slots[i].V - 1;
        float *logits = calloc((size_t)slots[i].V, sizeof(float));
        float *hidden = calloc((size_t)slots[i].E, sizeof(float));
        if (!logits || !hidden) { fprintf(stderr, "oom\n"); return 1; }
        slots[i].vt->prefill(t, 4, logits, hidden);
        int finite = 1;
        for (int k = 0; k < slots[i].V; k++)
            if (!(logits[k] == logits[k])) { finite = 0; break; }
        printf("%s (finite=%s)\n", slots[i].name, finite ? "yes" : "NO");
        top5(logits, slots[i].V, "prefill");
        slots[i].vt->decode(t[3], 4, logits, hidden);
        top5(logits, slots[i].V, "decode");
        free(logits); free(hidden);
    }

    for (int i = 0; i < 3; i++) slots[i].vt->release();
    return 0;
}
