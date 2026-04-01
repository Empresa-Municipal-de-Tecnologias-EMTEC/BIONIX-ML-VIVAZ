/*
 * arcface_wasm.c — Inferência ArcFace para WebAssembly (Emscripten)
 *
 * Implementa o mesmo pipeline que arcface_model.mojo / arcface_trainer.mojo:
 *   RGB → grayscale → BlocoCNN (conv2d+ReLU+avgpool2x2) → proj linear → L2-norm → embedding
 *
 * API exportada:
 *   int   bionix_load_bundle(uint8_t* data, int len)   — carrega pesos do bundle
 *   int   bionix_load_gallery(uint8_t* data, int len)  — carrega galeria (arcface_gallery.bin)
 *   void  bionix_embed_rgba(uint8_t* rgba, int w, int h, float* out_emb) — embedding de imagem RGBA
 *   float bionix_cosine_sim(float* a, float* b)        — similaridade cosseno
 *   int   bionix_identify_rgba(uint8_t* rgba, int w, int h, float threshold, float* out_score)
 *                                                      — retorna índice da classe (-1 = desconhecido)
 *   int   bionix_gallery_size(void)
 *   const char* bionix_get_name(int idx)
 *   void* bionix_malloc(int n)                         — malloc exposto ao JS
 *   void  bionix_free(void* p)                         — free exposto ao JS
 *   int   bionix_is_ready(void)                        — 1 se bundle carregado
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <emscripten.h>

/* ── Parâmetros máximos de dimensão ─────────────────────────────────────── */
#define MAX_KERNELS    16
#define MAX_KERNEL_H    7
#define MAX_KERNEL_W    7
#define MAX_PATCH     128
#define MAX_CLASSES_STATIC 512
#define MAX_NAME_LEN  256

/* ── Estado global do modelo ────────────────────────────────────────────── */
static int   g_ready        = 0;
static int   g_num_kernels  = 0;
static int   g_kernel_h     = 0;
static int   g_kernel_w     = 0;
static int   g_patch_size   = 0;
static int   g_embed_dim    = 0;
static int   g_feat_dim     = 0;
static int   g_num_classes  = 0;

static float g_kernels[MAX_KERNELS][MAX_KERNEL_H * MAX_KERNEL_W];

/* Pesos grandes alocados dinamicamente no primeiro load */
static float* g_proj_peso  = NULL;   /* [feat_dim × embed_dim] */
static float* g_proj_bias  = NULL;   /* [embed_dim] */
static float* g_cls_peso   = NULL;   /* [embed_dim × num_classes] */
static float* g_cls_bias   = NULL;   /* [num_classes] */
static float* g_feats_buf  = NULL;   /* work buffer [feat_dim] — reutilizado por thread */

/* ── Estado da galeria ──────────────────────────────────────────────────── */
static int    g_gallery_n   = 0;
static float* g_gallery_embs = NULL;  /* [n × embed_dim] dinâmico */
static char   g_gallery_names[MAX_CLASSES_STATIC][MAX_NAME_LEN];

/* ── Helpers internos ──────────────────────────────────────────────────── */
static inline float _read_f32le(const uint8_t* p) {
    uint32_t bits = (uint32_t)p[0] | ((uint32_t)p[1]<<8) |
                    ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
    float v;
    memcpy(&v, &bits, 4);
    return v;
}

static inline uint16_t _read_u16le(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static inline uint32_t _read_u32le(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) |
           ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}

/* ── Forward pass CNN ──────────────────────────────────────────────────── */

/* Conv2d valid + ReLU: input[H×W] → output[(H-kh+1)×(W-kw+1)] */
static void _conv2d_valid_relu(const float* img, int h, int w,
                                const float* kernel, int kh, int kw,
                                float* out, int out_h, int out_w) {
    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            float sum = 0.0f;
            for (int ky = 0; ky < kh; ky++)
                for (int kx = 0; kx < kw; kx++)
                    sum += img[(y+ky)*w + (x+kx)] * kernel[ky*kw + kx];
            out[y*out_w + x] = sum > 0.0f ? sum : 0.0f;   /* ReLU */
        }
    }
}

/* AvgPool 2×2 stride 2: input[h×w] → output[(h/2)×(w/2)] */
static void _avgpool2x2(const float* in, int h, int w, float* out) {
    int oh = h/2, ow = w/2;
    for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++)
            out[y*ow + x] = 0.25f * (in[(y*2)*w + (x*2)] +
                                     in[(y*2)*w + (x*2+1)] +
                                     in[(y*2+1)*w + (x*2)] +
                                     in[(y*2+1)*w + (x*2+1)]);
}

/* ── Embedding ─────────────────────────────────────────────────────────── */

/*
 * Calcula embedding L2-normalizado de uma imagem RGB HxW.
 * Realiza resize bilinear para patch_size×patch_size internamente.
 * out_emb: buffer de g_embed_dim floats.
 */
static void _embed_rgb_pixels(const float* rgb_f, int w, int h, float* out_emb) {
    int ps  = g_patch_size;
    int kh  = g_kernel_h;
    int kw  = g_kernel_w;
    int ch  = ps - kh + 1;
    int cw  = ps - kw + 1;
    int ph  = ch / 2;
    int pw  = cw / 2;
    int D   = g_feat_dim;
    int E   = g_embed_dim;

    /* Buffers no heap — evita stack overflow para patch_size grandes */
    float* gray     = (float*)malloc(ps * ps * sizeof(float));
    float* conv_buf = (float*)malloc(ps * ps * sizeof(float));
    float* pool_buf = (float*)malloc((ps/2) * (ps/2) * sizeof(float));
    float* feats    = g_feats_buf;   /* pré-alocado em bionix_load_bundle */
    if (!gray || !conv_buf || !pool_buf || !feats) {
        free(gray); free(conv_buf); free(pool_buf);
        return;
    }

    /* Grayscale + resize bilinear para ps×ps */
    float sx = (float)(w) / ps;
    float sy = (float)(h) / ps;
    for (int py = 0; py < ps; py++) {
        for (int px = 0; px < ps; px++) {
            float fx = (px + 0.5f) * sx - 0.5f;
            float fy = (py + 0.5f) * sy - 0.5f;
            int x0 = (int)fx; if (x0 < 0) x0 = 0; if (x0 >= w-1) x0 = w-2;
            int y0 = (int)fy; if (y0 < 0) y0 = 0; if (y0 >= h-1) y0 = h-2;
            float dx = fx - x0, dy = fy - y0;
            /* Bilinear interpolation across 4 neighbours */
            float r = (rgb_f[(y0*w+x0)*3+0]*(1-dx)*(1-dy) +
                       rgb_f[(y0*w+(x0+1))*3+0]*dx*(1-dy) +
                       rgb_f[((y0+1)*w+x0)*3+0]*(1-dx)*dy +
                       rgb_f[((y0+1)*w+(x0+1))*3+0]*dx*dy);
            float g = (rgb_f[(y0*w+x0)*3+1]*(1-dx)*(1-dy) +
                       rgb_f[(y0*w+(x0+1))*3+1]*dx*(1-dy) +
                       rgb_f[((y0+1)*w+x0)*3+1]*(1-dx)*dy +
                       rgb_f[((y0+1)*w+(x0+1))*3+1]*dx*dy);
            float b = (rgb_f[(y0*w+x0)*3+2]*(1-dx)*(1-dy) +
                       rgb_f[(y0*w+(x0+1))*3+2]*dx*(1-dy) +
                       rgb_f[((y0+1)*w+x0)*3+2]*(1-dx)*dy +
                       rgb_f[((y0+1)*w+(x0+1))*3+2]*dx*dy);
            gray[py*ps + px] = 0.299f*r + 0.587f*g + 0.114f*b;
        }
    }

    /* BlocoCNN: para cada filtro → conv+ReLU+pool → concatenar features */
    int off = 0;
    for (int f = 0; f < g_num_kernels; f++) {
        _conv2d_valid_relu(gray, ps, ps, g_kernels[f], kh, kw, conv_buf, ch, cw);
        _avgpool2x2(conv_buf, ch, cw, pool_buf);
        int pool_n = ph * pw;
        memcpy(&feats[off], pool_buf, pool_n * sizeof(float));
        off += pool_n;
    }

    /* Projeção linear [D → E] — proj reutiliza out_emb como buffer temporário */
    for (int e = 0; e < E; e++) {
        float s = g_proj_bias[e];
        for (int d = 0; d < D; d++)
            s += feats[d] * g_proj_peso[d * E + e];
        out_emb[e] = s;
    }

    /* L2-normalização in-place */
    float norm = 1e-8f;
    for (int e = 0; e < E; e++) norm += out_emb[e] * out_emb[e];
    norm = sqrtf(norm);
    for (int e = 0; e < E; e++) out_emb[e] /= norm;

    free(gray); free(conv_buf); free(pool_buf);
    /* feats aponta para g_feats_buf — não liberar aqui */
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  API pública exportada para JavaScript
 * ═══════════════════════════════════════════════════════════════════════════ */

EMSCRIPTEN_KEEPALIVE
int bionix_is_ready(void) { return g_ready; }

EMSCRIPTEN_KEEPALIVE
void* bionix_malloc(int n) { return malloc(n); }

EMSCRIPTEN_KEEPALIVE
void bionix_free(void* p) { free(p); }

/* Carrega bundle de pesos (arcface_bundle.bin). Retorna 1 em sucesso. */
EMSCRIPTEN_KEEPALIVE
int bionix_load_bundle(const uint8_t* data, int len) {
    if (len < 24) return 0;
    if (data[0]!='B'||data[1]!='N'||data[2]!='F'||data[3]!='X') return 0;

    g_num_kernels = data[5];
    g_kernel_h    = data[6];
    g_kernel_w    = data[7];
    g_patch_size  = (int)_read_u16le(&data[8]);
    g_embed_dim   = (int)_read_u16le(&data[10]);
    g_feat_dim    = (int)_read_u32le(&data[12]);
    g_num_classes = (int)_read_u32le(&data[16]);

    /* Aloca / re-aloca pesos dinâmicos */
    free(g_proj_peso); free(g_proj_bias); free(g_cls_peso); free(g_cls_bias); free(g_feats_buf);
    g_proj_peso = (float*)malloc(g_feat_dim * g_embed_dim * sizeof(float));
    g_proj_bias = (float*)malloc(g_embed_dim * sizeof(float));
    g_cls_peso  = (float*)malloc(g_embed_dim * g_num_classes * sizeof(float));
    g_cls_bias  = (float*)malloc(g_num_classes * sizeof(float));
    g_feats_buf = (float*)malloc(g_feat_dim * sizeof(float));
    if (!g_proj_peso || !g_proj_bias || !g_feats_buf) return 0;

    int offset = 24;
    int ksize  = g_kernel_h * g_kernel_w;

    /* Kernels */
    for (int k = 0; k < g_num_kernels; k++) {
        if (offset + ksize*4 > len) return 0;
        for (int i = 0; i < ksize; i++) {
            g_kernels[k][i] = _read_f32le(&data[offset]);
            offset += 4;
        }
    }
    /* proj_peso [D × E] */
    int pw_n = g_feat_dim * g_embed_dim;
    if (offset + pw_n*4 > len) return 0;
    for (int i = 0; i < pw_n; i++) {
        g_proj_peso[i] = _read_f32le(&data[offset]); offset += 4;
    }
    /* proj_bias [E] */
    for (int i = 0; i < g_embed_dim; i++) {
        g_proj_bias[i] = _read_f32le(&data[offset]); offset += 4;
    }
    /* cls_peso [E × C] */
    int cp_n = g_embed_dim * g_num_classes;
    if (offset + cp_n*4 > len) return 0;
    for (int i = 0; i < cp_n; i++) {
        g_cls_peso[i] = _read_f32le(&data[offset]); offset += 4;
    }
    /* cls_bias [C] */
    for (int i = 0; i < g_num_classes; i++) {
        g_cls_bias[i] = _read_f32le(&data[offset]); offset += 4;
    }

    g_ready = 1;
    return 1;
}

/* Carrega galeria (arcface_gallery.bin). Retorna número de classes carregadas. */
EMSCRIPTEN_KEEPALIVE
int bionix_load_gallery(const uint8_t* data, int len) {
    if (len < 8) return 0;
    int n = (int)_read_u32le(&data[0]);
    int E = (int)_read_u32le(&data[4]);
    if (n <= 0 || n >= MAX_CLASSES_STATIC || E != g_embed_dim) return 0;
    free(g_gallery_embs);
    g_gallery_embs = (float*)malloc(n * E * sizeof(float));
    if (!g_gallery_embs) return 0;
    int off = 8;
    for (int i = 0; i < n; i++) {
        if (off >= len) break;
        int nl = (int)data[off++];
        if (nl >= MAX_NAME_LEN) nl = MAX_NAME_LEN - 1;
        if (off + nl > len) break;
        memcpy(g_gallery_names[i], &data[off], nl);
        g_gallery_names[i][nl] = '\0';
        off += nl;
        if (off + E*4 > len) break;
        for (int e = 0; e < E; e++) {
            g_gallery_embs[i*E + e] = _read_f32le(&data[off]);
            off += 4;
        }
    }
    g_gallery_n = n;
    return n;
}

/* Calcula embedding de imagem RGBA (w×h, 4 bytes/pixel). Resultado em out_emb[embed_dim]. */
EMSCRIPTEN_KEEPALIVE
void bionix_embed_rgba(const uint8_t* rgba, int w, int h, float* out_emb) {
    if (!g_ready || w <= 0 || h <= 0 || !rgba || !out_emb) return;
    /* Convert RGBA→RGB floats [0,1] */
    float* rgb_f = (float*)malloc(w * h * 3 * sizeof(float));
    if (!rgb_f) return;
    for (int i = 0; i < w*h; i++) {
        rgb_f[i*3+0] = rgba[i*4+0] / 255.0f;
        rgb_f[i*3+1] = rgba[i*4+1] / 255.0f;
        rgb_f[i*3+2] = rgba[i*4+2] / 255.0f;
    }
    _embed_rgb_pixels(rgb_f, w, h, out_emb);
    free(rgb_f);
}

/* Similaridade cosseno entre dois embeddings já normalizados. */
EMSCRIPTEN_KEEPALIVE
float bionix_cosine_sim(const float* a, const float* b) {
    float s = 0.0f;
    for (int i = 0; i < g_embed_dim; i++) s += a[i] * b[i];
    return s;
}

/* Identifica a classe na galeria com melhor similaridade.
 * Retorna índice (-1 se nenhum superar threshold).
 * *out_score é preenchido com a melhor similaridade.
 */
EMSCRIPTEN_KEEPALIVE
int bionix_identify_rgba(const uint8_t* rgba, int w, int h,
                          float threshold, float* out_score) {
    if (!g_ready || !rgba || w <= 0 || h <= 0) { if(out_score)*out_score=0.0f; return -1; }
    float* emb = (float*)malloc(g_embed_dim * sizeof(float));
    if (!emb) { if(out_score)*out_score=0.0f; return -1; }
    bionix_embed_rgba(rgba, w, h, emb);
    if (g_gallery_n == 0) { if(out_score)*out_score=0.0f; free(emb); return -1; }
    int   best_idx  = -1;
    float best_sim  = -1.0f;
    int   E = g_embed_dim;
    for (int i = 0; i < g_gallery_n; i++) {
        float s = bionix_cosine_sim(emb, &g_gallery_embs[i * E]);
        if (s > best_sim) { best_sim = s; best_idx = i; }
    }
    free(emb);
    if (out_score) *out_score = best_sim;
    return (best_sim >= threshold) ? best_idx : -1;
}

EMSCRIPTEN_KEEPALIVE int   bionix_gallery_size(void) { return g_gallery_n; }
EMSCRIPTEN_KEEPALIVE int   bionix_embed_dim(void)    { return g_embed_dim;  }

EMSCRIPTEN_KEEPALIVE
const char* bionix_get_name(int idx) {
    if (idx < 0 || idx >= g_gallery_n) return "desconhecido";
    return g_gallery_names[idx];
}

/*
 * Detecta presença de rosto usando um threshold baixo contra qualquer entrada
 * da galeria. Retorna 1 se houver match mínimo, 0 caso contrário.
 * Parâmetro `presence_threshold` — use ~0.2 para presença geral de rosto.
 */
EMSCRIPTEN_KEEPALIVE
int bionix_has_face_rgba(const uint8_t* rgba, int w, int h, float presence_threshold) {
    if (!g_ready || !rgba || w <= 0 || h <= 0 || g_gallery_n == 0) return 0;
    /* embed_dim tipicamente 128; aloca no heap para evitar stack overflow */
    float* emb = (float*)malloc(g_embed_dim * sizeof(float));
    if (!emb) return 0;
    bionix_embed_rgba(rgba, w, h, emb);
    int   E = g_embed_dim;
    float best = -1.0f;
    for (int i = 0; i < g_gallery_n; i++) {
        float s = bionix_cosine_sim(emb, &g_gallery_embs[i * E]);
        if (s > best) best = s;
    }
    free(emb);
    return (best >= presence_threshold) ? 1 : 0;
}
