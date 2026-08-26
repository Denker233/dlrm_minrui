/*
 * libcoldread - optimised io_uring cold-embedding reader, callable from Python.
 *
 * Persistent worker threads, one io_uring per thread, O_DIRECT sector reads,
 * registered files, NUMA-pinned to the device each thread serves, optional
 * striping across two devices. Same engine as bench_iouring_opt.c, exposed as
 * cold_read() so a real DLRM forward pass can call it once per batch.
 *
 * build: gcc -O2 -shared -fPIC -o libcoldread.so libcoldread.c -luring -lpthread
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <liburing.h>

#define SECTOR 512
#define MAXT   32

static int g_nt, g_qd, g_ndev, g_row_bytes, g_ready = 0;
static int g_fds[2];
static pthread_t g_th[MAXT];
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv_go = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_cv_done = PTHREAD_COND_INITIALIZER;
static long g_gen = 0, g_done_cnt = 0;
static int g_stop = 0;
static const long *g_offs;
static int g_n;
static char *g_out;

typedef struct { int tid; char *buf; size_t cap; } wctx_t;

static void pin_to_node(int slot, int node) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
    FILE *f = fopen(path, "r");
    if (!f) return;
    int cpus[64], nc = 0, a, b;
    while (nc < 64 && fscanf(f, "%d-%d", &a, &b) == 2) {
        for (int c = a; c <= b && nc < 64; c++) cpus[nc++] = c;
        if (fgetc(f) != ',') break;
    }
    fclose(f);
    if (!nc) return;
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpus[slot % nc], &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

static void *worker(void *arg) {
    wctx_t *w = (wctx_t *)arg;
    int dev = w->tid % g_ndev;
    pin_to_node(w->tid / g_ndev, dev);

    struct io_uring ring;
    struct io_uring_params p;
    memset(&p, 0, sizeof(p));
    p.flags |= IORING_SETUP_IOPOLL;
    if (io_uring_queue_init_params(g_qd * 2, &ring, &p) < 0) return NULL;
    io_uring_register_files(&ring, g_fds, g_ndev);

    long seen = 0;
    for (;;) {
        pthread_mutex_lock(&g_mu);
        while (!g_stop && g_gen == seen) pthread_cond_wait(&g_cv_go, &g_mu);
        if (g_stop) { pthread_mutex_unlock(&g_mu); break; }
        seen = g_gen;
        pthread_mutex_unlock(&g_mu);

        int per = (g_n + g_nt - 1) / g_nt;
        int lo = w->tid * per;
        int hi = lo + per;
        if (lo > g_n) lo = g_n;
        if (hi > g_n) hi = g_n;
        int n = hi - lo;

        if (n > 0) {
            size_t need = (size_t)n * SECTOR;
            if (need > w->cap) {
                free(w->buf);
                if (posix_memalign((void **)&w->buf, 4096, need)) w->buf = NULL;
                w->cap = w->buf ? need : 0;
            }
            if (w->buf) {
                int sub = 0, done = 0;
                while (done < n) {
                    while (sub < n && (sub - done) < g_qd) {
                        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
                        if (!sqe) break;
                        long off = g_offs[lo + sub] & ~(long)(SECTOR - 1);
                        io_uring_prep_read(sqe, dev, w->buf + (size_t)sub * SECTOR, SECTOR, off);
                        sqe->flags |= IOSQE_FIXED_FILE;
                        sub++;
                    }
                    io_uring_submit(&ring);
                    struct io_uring_cqe *cqe;
                    if (io_uring_wait_cqe(&ring, &cqe) < 0) break;
                    io_uring_cqe_seen(&ring, cqe);
                    done++;
                    while (io_uring_peek_cqe(&ring, &cqe) == 0) {
                        io_uring_cqe_seen(&ring, cqe);
                        done++;
                    }
                }
                for (int i = 0; i < n; i++) {
                    long off = g_offs[lo + i];
                    long in_sector = off & (long)(SECTOR - 1);
                    memcpy(g_out + (size_t)(lo + i) * g_row_bytes,
                           w->buf + (size_t)i * SECTOR + in_sector,
                           g_row_bytes);
                }
            }
        }
        pthread_mutex_lock(&g_mu);
        if (++g_done_cnt == g_nt) pthread_cond_signal(&g_cv_done);
        pthread_mutex_unlock(&g_mu);
    }
    io_uring_queue_exit(&ring);
    free(w->buf);
    free(w);
    return NULL;
}

int cold_open(const char *dev1, const char *dev2, int nthreads, int qd, int row_bytes) {
    if (g_ready) return 0;
    g_nt = nthreads > MAXT ? MAXT : nthreads;
    g_qd = qd;
    g_row_bytes = row_bytes;
    g_ndev = (dev2 && dev2[0]) ? 2 : 1;
    g_fds[0] = open(dev1, O_RDONLY | O_DIRECT);
    if (g_fds[0] < 0) return -1;
    if (g_ndev == 2) {
        g_fds[1] = open(dev2, O_RDONLY | O_DIRECT);
        if (g_fds[1] < 0) return -2;
    }
    g_stop = 0; g_gen = 0; g_done_cnt = 0;
    for (int i = 0; i < g_nt; i++) {
        wctx_t *w = calloc(1, sizeof(wctx_t));
        w->tid = i;
        pthread_create(&g_th[i], NULL, worker, w);
    }
    g_ready = 1;
    return 0;
}

void cold_read(const long *offsets, int n, char *out) {
    pthread_mutex_lock(&g_mu);
    g_offs = offsets; g_n = n; g_out = out; g_done_cnt = 0; g_gen++;
    pthread_cond_broadcast(&g_cv_go);
    while (g_done_cnt < g_nt) pthread_cond_wait(&g_cv_done, &g_mu);
    pthread_mutex_unlock(&g_mu);
}

void cold_close(void) {
    pthread_mutex_lock(&g_mu);
    g_stop = 1;
    pthread_cond_broadcast(&g_cv_go);
    pthread_mutex_unlock(&g_mu);
    for (int i = 0; i < g_nt; i++) pthread_join(g_th[i], NULL);
    for (int d = 0; d < g_ndev; d++) close(g_fds[d]);
    g_ready = 0;
}
