/*
 * Optimised io_uring cold-embedding read benchmark — the strongest NVMe baseline
 * this hardware can produce, for a fair comparison against DC block-mean.
 *
 * Over the original bench_iouring.c this adds:
 *   - O_DIRECT + 512 B sector reads (device LBA format is 512 B) instead of
 *     buffered 4 KB page reads: 8x less read amplification, no page-cache copy
 *   - IORING_SETUP_IOPOLL: busy-polled completions, no interrupt latency
 *   - IORING_SETUP_SQPOLL (optional): kernel-side submission, zero syscalls
 *   - registered files + registered buffers: no per-op fd lookup or page pinning
 *   - N worker threads, each with its own ring, pinned to the NUMA node of the
 *     device it reads (nvme0n1 -> node 0, nvme1n1 -> node 1 on this box)
 *   - optional striping across both NVMe drives for 2x device parallelism
 *
 * Usage: bench_iouring_opt <offsets> <n> <threads> <qd> <rounds> <dev1> [dev2]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <liburing.h>

#define ROW_BYTES  64
#define SECTOR     512

static int      g_qd, g_rounds, g_nthreads, g_ndev, g_use_sqpoll, g_use_iopoll;
static long    *g_offs;
static int      g_n;
static int      g_fds[2];
static int      g_cpus_node[2][24];   /* cpu ids per numa node */
static int      g_ncpu_node[2];
static pthread_barrier_t g_barrier;

typedef struct { int tid, lo, hi; double ms; double *times; } worker_t;

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static void *worker(void *arg) {
    worker_t *w = (worker_t *)arg;
    int n = w->hi - w->lo;
    int dev = w->tid % g_ndev;              /* stripe threads across devices */

    /* pin to a cpu on this device's numa node */
    int node = dev;                          /* nvme0n1->0, nvme1n1->1 */
    if (g_ncpu_node[node] > 0) {
        cpu_set_t set; CPU_ZERO(&set);
        CPU_SET(g_cpus_node[node][(w->tid / g_ndev) % g_ncpu_node[node]], &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }

    struct io_uring ring;
    struct io_uring_params p;
    memset(&p, 0, sizeof(p));
    if (g_use_iopoll) p.flags |= IORING_SETUP_IOPOLL;
    if (g_use_sqpoll) { p.flags |= IORING_SETUP_SQPOLL; p.sq_thread_idle = 2000; }
    if (io_uring_queue_init_params(g_qd * 2, &ring, &p) < 0) {
        fprintf(stderr, "ring init failed (thread %d)\n", w->tid); w->ms = -1; return NULL;
    }
    io_uring_register_files(&ring, g_fds, g_ndev);

    void *buf;
    if (posix_memalign(&buf, 4096, (size_t)n * SECTOR)) { w->ms = -1; return NULL; }
    struct iovec iov = { .iov_base = buf, .iov_len = (size_t)n * SECTOR };
    io_uring_register_buffers(&ring, &iov, 1);

    double best = 1e18;
    for (int r = 0; r < g_rounds; r++) {
        pthread_barrier_wait(&g_barrier);
        double t0 = now_ms();
        int sub = 0, done = 0;
        while (done < n) {
            while (sub < n && (sub - done) < g_qd) {
                struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
                if (!sqe) break;
                long off = g_offs[w->lo + sub] & ~(long)(SECTOR - 1);  /* sector align */
                io_uring_prep_read_fixed(sqe, dev, (char *)buf + (size_t)sub * SECTOR,
                                         SECTOR, off, 0);
                sqe->flags |= IOSQE_FIXED_FILE;
                sub++;
            }
            io_uring_submit(&ring);
            struct io_uring_cqe *cqe;
            if (io_uring_wait_cqe(&ring, &cqe) < 0) break;
            io_uring_cqe_seen(&ring, cqe); done++;
            while (io_uring_peek_cqe(&ring, &cqe) == 0) { io_uring_cqe_seen(&ring, cqe); done++; }
        }
        double el = now_ms() - t0;
        w->times[r] = el;
        if (el < best) best = el;
    }
    w->ms = best;
    io_uring_queue_exit(&ring);
    free(buf);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s <offsets> <n> <threads> <qd> <rounds> <dev1> [dev2]\n", argv[0]);
        return 1;
    }
    const char *offfile = argv[1];
    g_n        = atoi(argv[2]);
    g_nthreads = atoi(argv[3]);
    g_qd       = atoi(argv[4]);
    g_rounds   = atoi(argv[5]);
    g_ndev     = (argc >= 8) ? 2 : 1;
    g_use_iopoll = getenv("NO_IOPOLL") ? 0 : 1;
    g_use_sqpoll = getenv("SQPOLL")    ? 1 : 0;

    /* numa cpu lists */
    for (int nd = 0; nd < 2; nd++) {
        char path[128]; snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", nd);
        FILE *f = fopen(path, "r");
        g_ncpu_node[nd] = 0;
        if (f) { int a, b; while (fscanf(f, "%d-%d", &a, &b) == 2) {
                    for (int c = a; c <= b && g_ncpu_node[nd] < 24; c++) g_cpus_node[nd][g_ncpu_node[nd]++] = c;
                    if (fgetc(f) != ',') break; }
                 fclose(f); }
    }

    g_offs = malloc((size_t)g_n * sizeof(long));
    FILE *f = fopen(offfile, "rb");
    if (!f) { perror("offsets"); return 1; }
    if (fread(g_offs, sizeof(long), g_n, f) != (size_t)g_n) { fprintf(stderr, "short offsets read\n"); return 1; }
    fclose(f);

    for (int d = 0; d < g_ndev; d++) {
        g_fds[d] = open(argv[6 + d], O_RDONLY | O_DIRECT);
        if (g_fds[d] < 0) { perror(argv[6 + d]); return 1; }
    }

    pthread_barrier_init(&g_barrier, NULL, g_nthreads);
    pthread_t th[64]; worker_t w[64];
    int per = (g_n + g_nthreads - 1) / g_nthreads;
    for (int i = 0; i < g_nthreads; i++) {
        w[i].tid = i; w[i].lo = i * per;
        w[i].hi = (w[i].lo + per > g_n) ? g_n : w[i].lo + per;
        if (w[i].lo > g_n) w[i].lo = w[i].hi = g_n;
        w[i].times = calloc(g_rounds, sizeof(double));
        pthread_create(&th[i], NULL, worker, &w[i]);
    }
    for (int i = 0; i < g_nthreads; i++) pthread_join(th[i], NULL);
    /* per round, the batch latency is the slowest thread (all must finish) */
    double *batch = calloc(g_rounds, sizeof(double));
    for (int r = 0; r < g_rounds; r++) {
        double mx = 0;
        for (int i = 0; i < g_nthreads; i++) if (w[i].times[r] > mx) mx = w[i].times[r];
        batch[r] = mx;
    }
    int cmp(const void *a, const void *b) {
        double d = *(const double *)a - *(const double *)b;
        return d < 0 ? -1 : d > 0 ? 1 : 0;
    }
    qsort(batch, g_rounds, sizeof(double), cmp);
    double sum = 0; for (int r = 0; r < g_rounds; r++) sum += batch[r];
    double mean = sum / g_rounds;
    double p50  = batch[g_rounds / 2];
    double p99  = batch[(int)(g_rounds * 0.99) >= g_rounds ? g_rounds - 1 : (int)(g_rounds * 0.99)];
    printf("%-8d %-6d %-8d %8.2f %8.2f %8.2f %8.2f %11.0f\n",
           g_nthreads, g_qd, g_ndev, batch[0], mean, p50, p99, g_n / (mean / 1000.0));
    for (int d = 0; d < g_ndev; d++) close(g_fds[d]);
    return 0;
}
