/*
 * io_uring batched read benchmark for cold embedding access.
 * Tests different queue depths on the same SATA SSD.
 *
 * Usage: ./bench_iouring <cold_file> <offsets_file> <num_offsets> <queue_depth> <num_rounds>
 *
 * offsets_file: binary file of int64_t byte offsets to read
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <liburing.h>

#define ROW_BYTES 64
#define PAGE_SIZE 4096

static double time_ms(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 +
           (end->tv_nsec - start->tv_nsec) / 1000000.0;
}


/* Evict THIS file's pages only.  The original used a global
 * `echo 3 > /proc/sys/vm/drop_caches`, which on a big-memory box also blows away
 * the page cache of anything else running (here: a 19h Terabyte preprocessing job).
 * posix_fadvise(DONTNEED) is surgical and much faster.  Set BENCH_DROP=global to
 * restore the original behaviour. */
static int drop_mode_global = -1;
static void drop_cache(int fd) {
    if (drop_mode_global < 0) {
        const char *e = getenv("BENCH_DROP");
        drop_mode_global = (e && strcmp(e, "global") == 0) ? 1 : 0;
    }
    if (drop_mode_global) {
        if (system("sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null")) { /* ignore */ }
    } else {
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    }
}

/* Strategy 1: Serial reads (QD=1 baseline) */
static double bench_serial(int fd, long *offsets, int n, char *buf) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < n; i++) {
        pread(fd, buf + i * ROW_BYTES, ROW_BYTES, offsets[i]);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return time_ms(&t0, &t1);
}

/* Strategy 2: io_uring batched reads at given queue depth */
static double bench_iouring(int fd, long *offsets, int n, int qd, char *buf) {
    struct io_uring ring;
    int ret = io_uring_queue_init(qd * 2, &ring, 0);
    if (ret < 0) {
        fprintf(stderr, "io_uring_queue_init failed: %d\n", ret);
        return -1;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int submitted = 0;
    int completed = 0;

    while (completed < n) {
        /* Submit up to qd requests */
        while (submitted < n && (submitted - completed) < qd) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            if (!sqe) break;
            io_uring_prep_read(sqe, fd, buf + submitted * ROW_BYTES,
                              ROW_BYTES, offsets[submitted]);
            io_uring_sqe_set_data(sqe, (void*)(long)submitted);
            submitted++;
        }
        io_uring_submit(&ring);

        /* Reap completions */
        struct io_uring_cqe *cqe;
        ret = io_uring_wait_cqe(&ring, &cqe);
        if (ret < 0) break;
        io_uring_cqe_seen(&ring, cqe);
        completed++;

        /* Reap any additional ready completions */
        while (io_uring_peek_cqe(&ring, &cqe) == 0) {
            io_uring_cqe_seen(&ring, cqe);
            completed++;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    io_uring_queue_exit(&ring);
    return time_ms(&t0, &t1);
}

/* Strategy 3: io_uring with page-aligned reads */
static double bench_iouring_pages(int fd, long *offsets, int n, int qd, char *pagebuf, int *n_pages_out) {
    /* Deduplicate by page */
    long *pages = malloc(n * sizeof(long));
    int np = 0;
    for (int i = 0; i < n; i++) {
        long page = (offsets[i] / PAGE_SIZE) * PAGE_SIZE;
        /* Simple dedup - check if already in list */
        int found = 0;
        for (int j = np - 1; j >= 0 && j >= np - 64; j--) {
            if (pages[j] == page) { found = 1; break; }
        }
        if (!found) pages[np++] = page;
    }
    *n_pages_out = np;

    struct io_uring ring;
    io_uring_queue_init(qd * 2, &ring, 0);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int submitted = 0;
    int completed = 0;

    while (completed < np) {
        while (submitted < np && (submitted - completed) < qd) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            if (!sqe) break;
            io_uring_prep_read(sqe, fd, pagebuf + submitted * PAGE_SIZE,
                              PAGE_SIZE, pages[submitted]);
            submitted++;
        }
        io_uring_submit(&ring);

        struct io_uring_cqe *cqe;
        io_uring_wait_cqe(&ring, &cqe);
        io_uring_cqe_seen(&ring, cqe);
        completed++;
        while (io_uring_peek_cqe(&ring, &cqe) == 0) {
            io_uring_cqe_seen(&ring, cqe);
            completed++;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    io_uring_queue_exit(&ring);
    free(pages);
    return time_ms(&t0, &t1);
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <cold_file> <offsets_file> <num_offsets> <queue_depth> <num_rounds>\n", argv[0]);
        return 1;
    }

    const char *cold_file = argv[1];
    const char *offsets_file = argv[2];
    int n = atoi(argv[3]);
    int max_qd = atoi(argv[4]);
    int rounds = atoi(argv[5]);

    /* Load offsets */
    FILE *f = fopen(offsets_file, "rb");
    long *offsets = malloc(n * sizeof(long));
    fread(offsets, sizeof(long), n, f);
    fclose(f);

    /* Open cold file with O_DIRECT */
    int fd = open(cold_file, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }

    /* Allocate aligned buffers */
    char *buf = aligned_alloc(4096, n * ROW_BYTES + 4096);
    char *pagebuf = aligned_alloc(4096, n * PAGE_SIZE + 4096);

    printf("=== io_uring Batched Read Benchmark ===\n");
    printf("File: %s\n", cold_file);
    printf("Offsets: %d reads of %d bytes each\n", n, ROW_BYTES);
    printf("Rounds: %d\n\n", rounds);

    /* Drop caches */
    drop_cache(fd);

    /* Serial baseline */
    printf("--- Serial pread (QD=1) ---\n");
    double serial_times[100];
    for (int r = 0; r < rounds; r++) {
        drop_cache(fd);
        serial_times[r] = bench_serial(fd, offsets, n, buf);
    }
    double serial_mean = 0;
    for (int r = 0; r < rounds; r++) serial_mean += serial_times[r];
    serial_mean /= rounds;
    printf("  mean=%.1fms\n", serial_mean);

    /* io_uring at different queue depths */
    int qds[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    int nqds = sizeof(qds) / sizeof(qds[0]);

    printf("\n--- io_uring row reads (64B) ---\n");
    printf("  %6s %10s %10s %10s\n", "QD", "Mean(ms)", "Speedup", "Eff.IOPS");
    printf("  %s\n", "----------------------------------------------");
    for (int qi = 0; qi < nqds; qi++) {
        int qd = qds[qi];
        if (qd > max_qd) break;
        double times[100];
        for (int r = 0; r < rounds; r++) {
            drop_cache(fd);
            times[r] = bench_iouring(fd, offsets, n, qd, buf);
        }
        double mean = 0;
        for (int r = 0; r < rounds; r++) mean += times[r];
        mean /= rounds;
        double iops = n / (mean / 1000.0);
        printf("  QD=%3d %9.1fms %9.2fx %9.0f\n", qd, mean, serial_mean / mean, iops);
    }

    /* io_uring with page-aligned reads */
    printf("\n--- io_uring page reads (4KB) ---\n");
    printf("  %6s %10s %10s %10s %8s\n", "QD", "Mean(ms)", "Speedup", "Eff.IOPS", "Pages");
    printf("  %s\n", "------------------------------------------------------");
    for (int qi = 0; qi < nqds; qi++) {
        int qd = qds[qi];
        if (qd > max_qd) break;
        double times[100];
        int np = 0;
        for (int r = 0; r < rounds; r++) {
            drop_cache(fd);
            times[r] = bench_iouring_pages(fd, offsets, n, qd, pagebuf, &np);
        }
        double mean = 0;
        for (int r = 0; r < rounds; r++) mean += times[r];
        mean /= rounds;
        double iops = np / (mean / 1000.0);
        printf("  QD=%3d %9.1fms %9.2fx %9.0f %7d\n", qd, mean, serial_mean / mean, iops, np);
    }

    printf("\n--- Summary ---\n");
    printf("  Serial pread:    %.1fms (baseline)\n", serial_mean);
    printf("  Best io_uring:   see above\n");
    printf("  DC block-mean:   0ms (no I/O)\n");

    close(fd);
    free(buf);
    free(pagebuf);
    free(offsets);
    return 0;
}
