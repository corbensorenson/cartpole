#pragma once
/* Optional evaluation-only exact initial-state injection and transition capture.
 * Training builds omit this header and all file IO from the stepping loop.
 * Binary files are little endian; the supported x86-64 platforms share this ABI.
 */
#include <stdlib.h>

typedef struct SpatialEval {
    FILE* plan;
    FILE* trace;
    char directory[4096];
    uint32_t cases, index;
    int exhausted;
} SpatialEval;

static void sp_io_fail(const char* message) {
    fprintf(stderr, "spatial evaluation: %s\n", message);
    exit(1);
}

static void sp_io_read(void* dst, size_t bytes, FILE* f) {
    if (fread(dst, 1, bytes, f) != bytes) sp_io_fail("truncated plan");
}

static void sp_io_write(const void* src, size_t bytes, FILE* f) {
    if (fwrite(src, 1, bytes, f) != bytes) sp_io_fail("trace write failed");
}

static void sp_eval_init(SpatialEval* io, const char* plan, const char* directory) {
    if (sizeof(double) != 8 || sizeof(float) != 4) sp_io_fail("unsupported numeric ABI");
    io->plan = fopen(plan, "rb");
    if (!io->plan) sp_io_fail("cannot open initial-state plan");
    if (strlen(directory) >= sizeof(io->directory)) sp_io_fail("trace directory too long");
    strcpy(io->directory, directory);
    char magic[8]; uint32_t fields[4];
    sp_io_read(magic, 8, io->plan); sp_io_read(fields, sizeof(fields), io->plan);
    if (memcmp(magic, "SPPLAN01", 8) || fields[0] != 1 ||
        fields[1] != SPATIAL_LINKS || fields[2] != SP_STATE || fields[3] < 1 || fields[3] > 45)
        sp_io_fail("invalid initial-state plan header");
    io->cases = fields[3];
    io->index = 0; io->trace = NULL; io->exhausted = 0;
}

static int sp_eval_reset(SpatialEval* io, SpatialCore* s) {
    if (io->trace) sp_io_fail("reset before previous trace completed");
    if (io->index == io->cases) {
        if (fgetc(io->plan) != EOF) sp_io_fail("unexpected trailing plan bytes");
        io->exhausted = 1;
        return 0;
    }
    int64_t seed; double noise; uint32_t mode; double state[SP_STATE];
    sp_io_read(&seed, sizeof(seed), io->plan);
    sp_io_read(&noise, sizeof(noise), io->plan);
    sp_io_read(&mode, sizeof(mode), io->plan);
    sp_io_read(state, sizeof(state), io->plan);
    for (int i = 0; i < SP_STATE; ++i) if (!isfinite(state[i])) sp_io_fail("nonfinite initial state");
    sp_reset_state(s, state);
    char path[8192];
    snprintf(path, sizeof(path), "%s/episode%04u.sptrace", io->directory, io->index++);
    io->trace = fopen(path, "wbx");
    if (!io->trace) sp_io_fail("trace exists or cannot be created");
    uint32_t fields[4] = {1, SPATIAL_LINKS, SP_STATE, SP_OBS};
    sp_io_write("SPTRACE1", 8, io->trace);
    sp_io_write(fields, sizeof(fields), io->trace);
    sp_io_write(&seed, sizeof(seed), io->trace);
    sp_io_write(&noise, sizeof(noise), io->trace);
    sp_io_write(&mode, sizeof(mode), io->trace);
    sp_io_write(&s->native_steps, sizeof(s->native_steps), io->trace);
    sp_io_write(state, sizeof(state), io->trace);
    return 1;
}

static void sp_eval_record(SpatialEval* io, const SpatialCore* s,
        const float* observation, const float* action, int advanced, float reward) {
    if (!io->trace) sp_io_fail("transition without an open trace");
    double raw[2] = {(double)action[0] * 80., (double)action[1] * 80.};
    uint32_t flags = s->warning | (s->done << 1) | (s->success << 2) | (advanced << 3);
    uint32_t fields[4] = {(uint32_t)s->steps, (uint32_t)s->streak, (uint32_t)s->best, flags};
    sp_io_write(fields, sizeof(fields), io->trace);
    sp_io_write(&s->native_steps, sizeof(s->native_steps), io->trace);
    sp_io_write(&s->margin, sizeof(s->margin), io->trace);
    sp_io_write(&reward, sizeof(reward), io->trace);
    sp_io_write(observation, SP_OBS * sizeof(float), io->trace);
    sp_io_write(action, 2 * sizeof(float), io->trace);
    sp_io_write(raw, sizeof(raw), io->trace);
    sp_io_write(s->delivered, sizeof(s->delivered), io->trace);
    sp_io_write(s->state, sizeof(s->state), io->trace);
    sp_io_write(s->tilts, sizeof(s->tilts), io->trace);
    if (s->done) {
        if (fclose(io->trace)) sp_io_fail("trace close failed");
        io->trace = NULL;
    }
}

static void sp_eval_close(SpatialEval* io) {
    if (io->trace) fclose(io->trace);  /* Partial trace cannot pass the importer. */
    if (io->plan) fclose(io->plan);
    io->trace = io->plan = NULL;
}
