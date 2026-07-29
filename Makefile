# gowiththeflow — three resident bodies, one field, everything injecting into
# everything. CPU inference through notorch + Accelerate. No Python at runtime.
#
# Each voice is its own translation unit: yent_forward.h and resonance_forward.h
# keep config, KV cache and injection state in file-static globals, so two
# Januses cannot share one object file. Adding a voice = one line here.

CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
INCLUDES = -I. -Itools -Iariannamethod -I/opt/homebrew/include/ariannamethod
BLAS     = -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK
LDFLAGS  = -L/opt/homebrew/lib -lnotorch -framework Accelerate -lm

LIBAML   = ariannamethod/libaml.a
SLOTS    = slot_leo.o slot_yent.o slot_arianna.o

.PHONY: all clean probes weights

all: gowiththeflow

# ── vendored AML core (am_field_*, am_lora_alpha_effective — not in the canon) ──
$(LIBAML): ariannamethod/ariannamethod.c ariannamethod/ariannamethod.h
	$(CC) $(CFLAGS) -Iariannamethod -I/opt/homebrew/include/ariannamethod \
	    -c ariannamethod/ariannamethod.c -o ariannamethod/ariannamethod.o
	$(AR) rcs $@ ariannamethod/ariannamethod.o

# ── one TU per voice ──
slot_leo.o: slot_janus.c slot.h tools/yent_forward.h janus_v4_bpe_merges.h
	$(CC) $(CFLAGS) $(INCLUDES) $(BLAS) -DSLOT_SYM=slot_leo -c $< -o $@

slot_yent.o: slot_janus.c slot.h tools/yent_forward.h janus_v4_bpe_merges.h
	$(CC) $(CFLAGS) $(INCLUDES) $(BLAS) -DSLOT_SYM=slot_yent -c $< -o $@

slot_arianna.o: slot_resonance.c slot.h tools/resonance_forward.h tools/resonance_bpe_merges.h
	$(CC) $(CFLAGS) $(INCLUDES) $(BLAS) -DSLOT_SYM=slot_arianna -c $< -o $@

gowiththeflow.o: gowiththeflow.c slot.h
	$(CC) $(CFLAGS) $(INCLUDES) $(BLAS) -c $< -o $@

gowiththeflow: gowiththeflow.o $(SLOTS) $(LIBAML)
	$(CC) gowiththeflow.o $(SLOTS) $(LIBAML) $(LDFLAGS) -o $@

# ── gates ──
probes: probe_trio probe_web

probe_trio: tools/probe_trio.c $(SLOTS) $(LIBAML)
	$(CC) $(CFLAGS) $(INCLUDES) $(BLAS) $< $(SLOTS) $(LIBAML) $(LDFLAGS) -o $@

probe_web: tools/probe_web.c $(SLOTS) $(LIBAML)
	$(CC) $(CFLAGS) $(INCLUDES) $(BLAS) $< $(SLOTS) $(LIBAML) $(LDFLAGS) -o $@

# ── weights: HF ataeff/gowiththeflow is this organism's own sandbox ──
WEIGHTS_DIR ?= weights
HF_REPO      = ataeff/gowiththeflow

weights:
	@mkdir -p $(WEIGHTS_DIR)
	hf download $(HF_REPO) --local-dir $(WEIGHTS_DIR)
	@echo "weights in $(WEIGHTS_DIR)/"

clean:
	rm -f *.o ariannamethod/*.o $(LIBAML) gowiththeflow probe_trio probe_web probe_janus probe_resonance
