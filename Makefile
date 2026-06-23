CC      = /opt/rocm/llvm/bin/clang
CFLAGS  = -std=c11 -O2 -Wall -Wextra -Wpedantic \
          -Iinclude -Ivendor
LDFLAGS = -lm

LIB_SRCS = src/safetensors_loader.c \
           src/tensor.c

# Two binaries:
#   run      — llama2.c-style inference entry point (src/run.c)
#   mla-moe  — weight inspection CLI (src/main.c)
RUN_SRCS  = $(LIB_SRCS) src/run.c
TOOL_SRCS = $(LIB_SRCS) src/main.c

.PHONY: all clean

all: run mla-moe

run: $(RUN_SRCS)
	$(CC) $(CFLAGS) $(RUN_SRCS) -o $@ $(LDFLAGS)

mla-moe: $(TOOL_SRCS)
	$(CC) $(CFLAGS) $(TOOL_SRCS) -o $@ $(LDFLAGS)

debug-run: CFLAGS += -O0 -g -fsanitize=address,undefined
debug-run: $(RUN_SRCS)
	$(CC) $(CFLAGS) $(RUN_SRCS) -o run $(LDFLAGS)

debug-tool: CFLAGS += -O0 -g -fsanitize=address,undefined
debug-tool: $(TOOL_SRCS)
	$(CC) $(CFLAGS) $(TOOL_SRCS) -o mla-moe $(LDFLAGS)

clean:
	rm -f run mla-moe
