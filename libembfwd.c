/*
 * libembfwd - fused C embedding-gather paths, so fp32 / DC / SSD are all
 * measured with the SAME implementation technology.  Comparing a fused C DC
 * gather against a PyTorch fp32 gather would flatter DC; this removes that bias.
 *
 * Each fwd_*_all() does ALL tables for one batch in a single call:
 *   - one pass over the batch, no temporaries, values stay in registers
 *   - OpenMP across tables+rows
 *   - SSD path issues ONE io_uring batch covering every cold row in every table
 *
 * build: gcc -O2 -march=native -fopenmp -shared -fPIC -o libembfwd.so libembfwd.c -luring -lpthread
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <omp.h>
#include <liburing.h>

#define SECTOR 512
#define MAXT   32

/* ---------------- io_uring engine (same as libcoldread) ---------------- */
static int g_nt, g_qd, g_ndev, g_row_bytes, g_ready = 0;
static int g_fds[2];
static pthread_t g_th[MAXT];
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv_go = PTHREAD_COND_INITIALIZER, g_cv_done = PTHREAD_COND_INITIALIZER;
static long g_gen = 0, g_done_cnt = 0;
static int g_stop = 0;
static const long *g_offs; static int g_n; static char *g_out;
typedef struct { int tid; char *buf; size_t cap; } wctx_t;

static void pin_node(int slot, int node) {
    char p[128]; snprintf(p, sizeof(p), "/sys/devices/system/node/node%d/cpulist", node);
    FILE *f = fopen(p, "r"); if (!f) return;
    int c[64], nc = 0, a, b;
    while (nc < 64 && fscanf(f, "%d-%d", &a, &b) == 2) {
        for (int x = a; x <= b && nc < 64; x++) c[nc++] = x;
        if (fgetc(f) != ',') break;
    }
    fclose(f); if (!nc) return;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(c[slot % nc], &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

static void *ioworker(void *arg) {
    wctx_t *w = (wctx_t *)arg;
    int dev = w->tid % g_ndev;
    pin_node(w->tid / g_ndev, dev);
    struct io_uring ring; struct io_uring_params p; memset(&p, 0, sizeof(p));
    p.flags |= IORING_SETUP_IOPOLL;
    if (io_uring_queue_init_params(g_qd * 2, &ring, &p) < 0) return NULL;
    io_uring_register_files(&ring, g_fds, g_ndev);
    long seen = 0;
    for (;;) {
        pthread_mutex_lock(&g_mu);
        while (!g_stop && g_gen == seen) pthread_cond_wait(&g_cv_go, &g_mu);
        if (g_stop) { pthread_mutex_unlock(&g_mu); break; }
        seen = g_gen; pthread_mutex_unlock(&g_mu);
        int per = (g_n + g_nt - 1) / g_nt;
        int lo = w->tid * per, hi = lo + per;
        if (lo > g_n) lo = g_n;
        if (hi > g_n) hi = g_n;
        int n = hi - lo;
        if (n > 0) {
            size_t need = (size_t)n * SECTOR;
            if (need > w->cap) { free(w->buf);
                if (posix_memalign((void **)&w->buf, 4096, need)) w->buf = NULL;
                w->cap = w->buf ? need : 0; }
            if (w->buf) {
                int sub = 0, done = 0;
                while (done < n) {
                    while (sub < n && (sub - done) < g_qd) {
                        struct io_uring_sqe *s = io_uring_get_sqe(&ring); if (!s) break;
                        long off = g_offs[lo + sub] & ~(long)(SECTOR - 1);
                        io_uring_prep_read(s, dev, w->buf + (size_t)sub * SECTOR, SECTOR, off);
                        s->flags |= IOSQE_FIXED_FILE; sub++;
                    }
                    io_uring_submit(&ring);
                    struct io_uring_cqe *c;
                    if (io_uring_wait_cqe(&ring, &c) < 0) break;
                    io_uring_cqe_seen(&ring, c); done++;
                    while (io_uring_peek_cqe(&ring, &c) == 0) { io_uring_cqe_seen(&ring, c); done++; }
                }
                for (int i = 0; i < n; i++)
                    memcpy(g_out + (size_t)(lo + i) * g_row_bytes,
                           w->buf + (size_t)i * SECTOR + (g_offs[lo + i] & (SECTOR - 1)),
                           g_row_bytes);
            }
        }
        pthread_mutex_lock(&g_mu);
        if (++g_done_cnt == g_nt) pthread_cond_signal(&g_cv_done);
        pthread_mutex_unlock(&g_mu);
    }
    io_uring_queue_exit(&ring); free(w->buf); free(w); return NULL;
}

int emb_open(const char *d1, const char *d2, int nthreads, int qd, int row_bytes) {
    if (g_ready) return 0;
    g_nt = nthreads > MAXT ? MAXT : nthreads; g_qd = qd; g_row_bytes = row_bytes;
    g_ndev = (d2 && d2[0]) ? 2 : 1;
    g_fds[0] = open(d1, O_RDONLY | O_DIRECT); if (g_fds[0] < 0) return -1;
    if (g_ndev == 2) { g_fds[1] = open(d2, O_RDONLY | O_DIRECT); if (g_fds[1] < 0) return -2; }
    g_stop = 0; g_gen = 0; g_done_cnt = 0;
    for (int i = 0; i < g_nt; i++) { wctx_t *w = calloc(1, sizeof(wctx_t)); w->tid = i;
        pthread_create(&g_th[i], NULL, ioworker, w); }
    g_ready = 1; return 0;
}
static void do_read(const long *offs, int n, char *out) {
    pthread_mutex_lock(&g_mu);
    g_offs = offs; g_n = n; g_out = out; g_done_cnt = 0; g_gen++;
    pthread_cond_broadcast(&g_cv_go);
    while (g_done_cnt < g_nt) pthread_cond_wait(&g_cv_done, &g_mu);
    pthread_mutex_unlock(&g_mu);
}
void emb_close(void) {
    pthread_mutex_lock(&g_mu); g_stop = 1; pthread_cond_broadcast(&g_cv_go);
    pthread_mutex_unlock(&g_mu);
    for (int i = 0; i < g_nt; i++) pthread_join(g_th[i], NULL);
    for (int d = 0; d < g_ndev; d++) close(g_fds[d]);
    g_ready = 0;
}

/* ---------------- fused gathers ---------------- */

/* fp32 baseline: out[t][b][:] = W[t][idx[t][b]][:] */
void fwd_fp32_all(const long **idx, const float **W, int ntab, int bs, int D, float **out) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int t = 0; t < ntab; t++)
        for (int b = 0; b < bs; b++)
            memcpy(out[t] + (size_t)b * D, W[t] + (size_t)idx[t][b] * D, (size_t)D * 4);
}

/* DC: hot rows uint8 (dequantised inline), cold rows = 4-bit block mean broadcast */
void fwd_dc_all(const long **idx, const long **hot_pos, const unsigned char **hot_u8,
                const float *hs, const float *hmn, const long **cold_rank,
                const unsigned char **dcq, const float *ds, const float *dlo,
                const float **W, const int *is_large, int block,
                int ntab, int bs, int D, float **out) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int t = 0; t < ntab; t++) {
        for (int b = 0; b < bs; b++) {
            float *o = out[t] + (size_t)b * D;
            long r = idx[t][b];
            if (!is_large[t]) { memcpy(o, W[t] + (size_t)r * D, (size_t)D * 4); continue; }
            long p = hot_pos[t][r];
            if (p >= 0) {
                const unsigned char *src = hot_u8[t] + (size_t)p * D;
                float s = hs[t], m = hmn[t];
                for (int d = 0; d < D; d++) o[d] = (float)src[d] * s + m;
            } else {
                long blk = cold_rank[t][r] / block;
                float v = (float)dcq[t][blk] * ds[t] + dlo[t];
                for (int d = 0; d < D; d++) o[d] = v;
            }
        }
    }
}

/* SSD: hot rows uint8 from DRAM, cold rows exact fp32 from NVMe.
   One io_uring batch for every cold row across every table. */
static long  *s_offs = NULL; static long *s_slot = NULL; static char *s_buf = NULL;
static size_t s_cap = 0;
void fwd_ssd_all(const long **idx, const long **hot_pos, const unsigned char **hot_u8,
                 const float *hs, const float *hmn, const long *base,
                 const float **W, const int *is_large,
                 int ntab, int bs, int D, float **out) {
    int maxn = ntab * bs;
    if ((size_t)maxn > s_cap) {
        free(s_offs); free(s_slot); free(s_buf);
        s_offs = malloc((size_t)maxn * sizeof(long));
        s_slot = malloc((size_t)maxn * 2 * sizeof(long));
        s_buf  = malloc((size_t)maxn * D * 4);
        s_cap  = maxn;
    }
    /* pass 1: plan (serial, cheap - just index arithmetic) */
    int nc = 0;
    for (int t = 0; t < ntab; t++) {
        if (!is_large[t]) continue;
        for (int b = 0; b < bs; b++) {
            long r = idx[t][b];
            if (hot_pos[t][r] < 0) {
                s_offs[nc] = base[t] + r * (long)(D * 4);
                s_slot[2*nc] = t; s_slot[2*nc+1] = b;
                nc++;
            }
        }
    }
    if (nc) do_read(s_offs, nc, s_buf);
    /* pass 2: assemble hot rows in parallel */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int t = 0; t < ntab; t++) {
        for (int b = 0; b < bs; b++) {
            float *o = out[t] + (size_t)b * D;
            long r = idx[t][b];
            if (!is_large[t]) { memcpy(o, W[t] + (size_t)r * D, (size_t)D * 4); continue; }
            long p = hot_pos[t][r];
            if (p >= 0) {
                const unsigned char *src = hot_u8[t] + (size_t)p * D;
                float s = hs[t], m = hmn[t];
                for (int d = 0; d < D; d++) o[d] = (float)src[d] * s + m;
            }
        }
    }
    /* scatter the cold rows that came off disk */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nc; i++) {
        long t = s_slot[2*i], b = s_slot[2*i+1];
        memcpy(out[t] + (size_t)b * D, s_buf + (size_t)i * D * 4, (size_t)D * 4);
    }
}

/* whole-table INT8: uint8 rows, dequantised inline (the strongest competing baseline) */
void fwd_int8_all(const long **idx, const unsigned char **Q, const float *s, const float *mn,
                  const float **W, const int *is_large, int ntab, int bs, int D, float **out) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int t = 0; t < ntab; t++) {
        for (int b = 0; b < bs; b++) {
            float *o = out[t] + (size_t)b * D;
            long r = idx[t][b];
            if (!is_large[t]) { memcpy(o, W[t] + (size_t)r * D, (size_t)D * 4); continue; }
            const unsigned char *src = Q[t] + (size_t)r * D;
            float sc = s[t], m = mn[t];
            for (int d = 0; d < D; d++) o[d] = (float)src[d] * sc + m;
        }
    }
}


/* DC with PER-DIMENSION block means, 4-bit nibble-packed.
   cold row -> block = cold_rank/block; block holds D/2 bytes = D nibbles. */
void fwd_dcpd_all(const long **idx, const long **hot_pos, const unsigned char **hot_u8,
                  const float *hs, const float *hmn, const long **cold_rank,
                  const unsigned char **dcq, const float *ds, const float *dlo,
                  const float **W, const int *is_large, int block,
                  int ntab, int bs, int D, float **out) {
    const int half = D / 2;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int t = 0; t < ntab; t++) {
        for (int b = 0; b < bs; b++) {
            float *o = out[t] + (size_t)b * D;
            long r = idx[t][b];
            if (!is_large[t]) { memcpy(o, W[t] + (size_t)r * D, (size_t)D * 4); continue; }
            long p = hot_pos[t][r];
            if (p >= 0) {
                const unsigned char *src = hot_u8[t] + (size_t)p * D;
                float s = hs[t], m = hmn[t];
                for (int d = 0; d < D; d++) o[d] = (float)src[d] * s + m;
            } else {
                long blk = cold_rank[t][r] / block;
                const unsigned char *q = dcq[t] + (size_t)blk * half;
                float s = ds[t], m = dlo[t];
                for (int d = 0; d < half; d++) {
                    unsigned char byte = q[d];
                    o[2*d]   = (float)(byte >> 4)  * s + m;
                    o[2*d+1] = (float)(byte & 0xF) * s + m;
                }
            }
        }
    }
}


/* whole-table INT4: nibble-packed rows (D/2 bytes each), unpacked inline */
void fwd_int4_all(const long **idx, const unsigned char **Q, const float *s, const float *mn,
                  const float **W, const int *is_large, int ntab, int bs, int D, float **out) {
    const int half = D / 2;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int t = 0; t < ntab; t++) {
        for (int b = 0; b < bs; b++) {
            float *o = out[t] + (size_t)b * D;
            long r = idx[t][b];
            if (!is_large[t]) { memcpy(o, W[t] + (size_t)r * D, (size_t)D * 4); continue; }
            const unsigned char *q = Q[t] + (size_t)r * half;
            float sc = s[t], m = mn[t];
            for (int d = 0; d < half; d++) {
                unsigned char byte = q[d];
                o[2*d]   = (float)(byte >> 4)  * sc + m;
                o[2*d+1] = (float)(byte & 0xF) * sc + m;
            }
        }
    }
}

int last_cold_count(void) { return 0; }
