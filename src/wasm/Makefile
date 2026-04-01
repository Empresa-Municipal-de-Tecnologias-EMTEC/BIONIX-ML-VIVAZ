# Makefile – compila arcface_wasm.c com Emscripten → build/arcface.js + arcface.wasm
#
# Pré-requisito: emcc instalado (https://emscripten.org/docs/getting_started/downloads.html)
#   sudo apt install emscripten          (Ubuntu/Debian)
#   ou via emsdk: source emsdk/emsdk_env.sh
#
# Para compilar:
#   make           — release optimizado
#   make debug     — debug com asserts
#   make clean

OUTDIR   := ../build
OUTBASE  := $(OUTDIR)/arcface
SRC      := arcface_wasm.c

EXPORTS := '["_bionix_load_bundle","_bionix_load_gallery", \
              "_bionix_embed_rgba","_bionix_cosine_sim", \
              "_bionix_identify_rgba","_bionix_has_face_rgba", \
              "_bionix_gallery_size","_bionix_embed_dim", \
              "_bionix_get_name","_bionix_is_ready", \
              "_bionix_malloc","_bionix_free", \
              "_malloc","_free"]'

RUNTIME  := '["ccall","cwrap","getValue","setValue","UTF8ToString","HEAPU8","HEAPF32"]'

CFLAGS_COMMON := -I. \
  -s WASM=1 \
  -s EXPORTED_FUNCTIONS=$(EXPORTS) \
  -s EXPORTED_RUNTIME_METHODS=$(RUNTIME) \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=32MB \
  -s MAXIMUM_MEMORY=256MB \
  -s STACK_SIZE=262144 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME='BionixModule' \
  -s ENVIRONMENT='web' \
  -lm

all: $(OUTDIR)
	emcc $(SRC) -O2 $(CFLAGS_COMMON) -o $(OUTBASE).js
	@echo "✓ WASM compilado → $(OUTBASE).js + $(OUTBASE).wasm"

debug: $(OUTDIR)
	emcc $(SRC) -O0 -g $(CFLAGS_COMMON) -s ASSERTIONS=1 -o $(OUTBASE)_debug.js
	@echo "✓ WASM debug compilado → $(OUTBASE)_debug.js"

$(OUTDIR):
	mkdir -p $(OUTDIR)

clean:
	rm -f $(OUTBASE).js $(OUTBASE).wasm $(OUTBASE)_debug.js $(OUTBASE)_debug.wasm

.PHONY: all debug clean
