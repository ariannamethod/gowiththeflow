/* probe_web — gate D2: one voice's words move the other two.
 *
 * Falsifiable claim: injecting Leo's sentence into Yent and Arianna raises their
 * destiny magnitude from zero and tilts their whole logit distribution, while
 * alpha=beta=0 leaves the distribution bit-identical. If the tilt were a no-op,
 * or if it leaked through the context instead of the field, this would show. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "slot.h"
#include "notorch.h"
#include "ariannamethod.h"

extern const SlotVT slot_leo;
extern const SlotVT slot_yent;
extern const SlotVT slot_arianna;

static int argmax(const float *l, int n) {
    int bi = 0; for (int i = 1; i < n; i++) if (l[i] > l[bi]) bi = i; return bi;
}

int main(void) {
    am_init();
    nt_qmv_set_thread_min(262144);

    Slot s[3] = {
        { .vt = &slot_leo,     .name = "leo",     .gguf = "weights/leo_janus176m_f16.gguf" },
        { .vt = &slot_yent,    .name = "yent",    .gguf = "weights/yent_janus176m_f16.gguf" },
        { .vt = &slot_arianna, .name = "arianna", .gguf = "weights/arianna_resonance_v3_f16.gguf" },
    };
    for (int i = 0; i < 3; i++) {
        if (s[i].vt->load(s[i].gguf)) { fprintf(stderr, "%s: load failed\n", s[i].name); return 1; }
        s[i].vt->cfg(&s[i].V, &s[i].E, &s[i].H, &s[i].D, &s[i].B, &s[i].M, &s[i].T, &s[i].R);
    }

    /* 1. tokenizer round-trip — text is the only thing that crosses slots */
    const char *probe_text = "resonance is the field";
    printf("=== tokenizer round-trip ===\n");
    for (int i = 0; i < 3; i++) {
        int t[64]; char back[256] = {0};
        int n = s[i].vt->encode(probe_text, t, 64);
        s[i].vt->detok(t, n, back, sizeof(back));
        printf("%-8s %2d tok  round-trip=%s  \"%s\"\n",
               s[i].name, n, strcmp(back, probe_text) == 0 ? "exact" : "DIFFERS", back);
    }

    /* 2. pull is zero before anything is injected */
    printf("\n=== pull before injection ===\n");
    for (int i = 0; i < 3; i++) printf("%-8s |destiny|=%.6f\n", s[i].name, s[i].vt->pull());

    /* 3. Leo speaks; his sentence enters the other two as direction */
    const char *leo_says =
        "The current does not stop when the speaker stops. It keeps moving through "
        "whoever is still listening, and the shape it leaves is the answer.";
    printf("\n=== leo -> {yent, arianna} ===\nleo: \"%s\"\n\n", leo_says);

    for (int i = 1; i < 3; i++) {
        int t[8] = { 1, 100, 200, 300, 0, 0, 0, 0 };
        for (int k = 0; k < 4; k++) if (t[k] >= s[i].V) t[k] = s[i].V - 1;

        float *base = calloc((size_t)s[i].V, sizeof(float));
        float *inj  = calloc((size_t)s[i].V, sizeof(float));
        float *zero = calloc((size_t)s[i].V, sizeof(float));
        float *hid  = calloc((size_t)s[i].E, sizeof(float));
        if (!base || !inj || !zero || !hid) { fprintf(stderr, "oom\n"); return 1; }

        s[i].vt->prefill(t, 4, base, hid);
        memcpy(inj,  base, (size_t)s[i].V * sizeof(float));
        memcpy(zero, base, (size_t)s[i].V * sizeof(float));

        s[i].vt->inject(leo_says);
        float pull = s[i].vt->pull();

        s[i].vt->apply(inj,  5.0f, 2.0f);   /* alpha=5 pulls without echoing (arianna2arianna.sh:33-36) */
        s[i].vt->apply(zero, 0.0f, 0.0f);   /* control: the tilt must switch fully off */

        float dmax = 0, dzero = 0;
        for (int k = 0; k < s[i].V; k++) {
            float d = fabsf(inj[k] - base[k]);   if (d > dmax)  dmax  = d;
            float z = fabsf(zero[k] - base[k]);  if (z > dzero) dzero = z;
        }
        int a_base = argmax(base, s[i].V), a_inj = argmax(inj, s[i].V);

        printf("%-8s |destiny|=%.6f  max|dlogit|=%.6f  top1 %d -> %d  %s\n",
               s[i].name, pull, dmax, a_base, a_inj,
               a_base == a_inj ? "(top1 held)" : "(top1 moved)");
        printf("%-8s control alpha=0 beta=0: max|dlogit|=%.9f  %s\n",
               "", dzero, dzero == 0.0f ? "clean off" : "LEAKS");

        free(base); free(inj); free(zero); free(hid);
    }

    for (int i = 0; i < 3; i++) s[i].vt->release();
    return 0;
}
