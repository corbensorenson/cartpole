#pragma once
/* Native stepping core. No Python, allocation, or rendering in sp_step. */
#include <mujoco/mujoco.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef SPATIAL_LINKS
#define SPATIAL_LINKS 1
#endif
#define SP_NQ (2 + 4 * SPATIAL_LINKS)
#define SP_NV (2 + 3 * SPATIAL_LINKS)
#define SP_STATE (SP_NQ + SP_NV)
#define SP_OBS (SP_STATE + 3)

typedef struct SpatialCore {
    const mjModel* model;
    mjData* data;
    int link_ids[SPATIAL_LINKS];
    int steps, streak, best, warning, done, success;
    uint64_t native_steps;
    double margin, max_tilt, tilts[SPATIAL_LINKS];
    float delivered[2];
    double state[SP_STATE];
} SpatialCore;

static int sp_init(SpatialCore* s, const mjModel* model) {
    memset(s, 0, sizeof(*s));
    if (model->nq != SP_NQ || model->nv != SP_NV || model->nu != 2 ||
        model->opt.timestep != .005 || model->opt.integrator != mjINT_RK4) return 0;
    s->model = model;
    s->data = mj_makeData(model);
    if (!s->data) return 0;
    for (int i = 0; i < SPATIAL_LINKS; ++i) {
        char name[32];
        snprintf(name, sizeof(name), "link_%d", i);
        s->link_ids[i] = mj_name2id(model, mjOBJ_BODY, name);
        if (s->link_ids[i] < 0) { mj_deleteData(s->data); s->data = NULL; return 0; }
    }
    return 1;
}

static void sp_sync(SpatialCore* s) {
    memcpy(s->state, s->data->qpos, SP_NQ * sizeof(double));
    memcpy(s->state + SP_NQ, s->data->qvel, SP_NV * sizeof(double));
}

static void sp_set_state(SpatialCore* s, const double* state) {
    /* Alias safe. Reset integration scratch exactly as the authoritative plant. */
    double copy[SP_STATE];
    memcpy(copy, state, sizeof(copy));
    mj_resetData(s->model, s->data);
    memcpy(s->data->qpos, copy, SP_NQ * sizeof(double));
    memcpy(s->data->qvel, copy + SP_NQ, SP_NV * sizeof(double));
    mj_forward(s->model, s->data);
    sp_sync(s);
}

static void sp_update_tilts(SpatialCore* s) {
    s->max_tilt = 0;
    for (int i = 0; i < SPATIAL_LINKS; ++i) {
        double z = s->data->xmat[9 * s->link_ids[i] + 8];
        s->tilts[i] = acos(fmax(-1., fmin(1., z)));
        s->max_tilt = fmax(s->max_tilt, s->tilts[i]);
    }
}

static void sp_reset_state(SpatialCore* s, const double* state) {
    sp_set_state(s, state);
    s->steps = s->streak = s->best = s->warning = s->done = s->success = 0;
    s->margin = 3.;
    s->delivered[0] = s->delivered[1] = 0;
    sp_update_tilts(s);
}

static uint64_t sp_random(uint64_t* rng) {
    uint64_t z = (*rng += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static double sp_uniform(uint64_t* rng, double radius) {
    return (2. * (double)(sp_random(rng) >> 11) * (1. / 9007199254740992.) - 1.) * radius;
}

static void sp_reset_random(SpatialCore* s, uint64_t* rng, double noise) {
    double state[SP_STATE] = {0}, delta[SP_NV] = {0};
    memcpy(state, s->model->qpos0, SP_NQ * sizeof(double));
    state[2] = state[3] = state[5] = 0.; state[4] = 1.;
    if (noise > 0) {
        delta[0] = sp_uniform(rng, .05); delta[1] = sp_uniform(rng, .05);
        for (int j = 2; j < SP_NV; ++j) delta[j] = sp_uniform(rng, noise);
        for (int j = 0; j < SP_NV; ++j) state[SP_NQ + j] = sp_uniform(rng, 2. * noise);
        mj_integratePos(s->model, state, delta, 1.);
    }
    sp_reset_state(s, state);
}

static int sp_deliver(const double* raw, float* result) {
    if (!isfinite(raw[0]) || !isfinite(raw[1])) return 0;
    double norm = sqrt(raw[0] * raw[0] + raw[1] * raw[1]);
    double scale = fmin(1., 80. / fmax(norm, 1e-30));
    result[0] = (float)(raw[0] * scale); result[1] = (float)(raw[1] * scale);
    while (sqrt((double)result[0] * result[0] + (double)result[1] * result[1]) > 80.) {
        result[0] = nextafterf(result[0], 0.f);
        result[1] = nextafterf(result[1], 0.f);
    }
    return 1;
}

static int sp_step(SpatialCore* s, const double* raw) {
    if (s->done) return 0;
    if (!sp_deliver(raw, s->delivered)) { s->warning = s->done = 1; return 0; }
    sp_set_state(s, s->state);
    s->data->ctrl[0] = s->delivered[0]; s->data->ctrl[1] = s->delivered[1];
    for (int sub = 0; sub < 4; ++sub) {
        mj_step(s->model, s->data);
        s->native_steps++;
        double extent = fmax(fabs(s->data->qpos[0]), fabs(s->data->qpos[1]));
        s->margin = fmin(s->margin, 3. - extent);
        for (int w = 0; w < mjNWARNING; ++w) s->warning |= s->data->warning[w].number > 0;
    }
    mj_forward(s->model, s->data);
    sp_sync(s);
    s->steps++;
    sp_update_tilts(s);
    s->streak = s->max_tilt < .15 ? s->streak + 1 : 0;
    if (s->streak > s->best) s->best = s->streak;
    int finite = 1;
    for (int j = 0; j < SP_STATE; ++j) finite = finite && isfinite(s->state[j]);
    s->done = s->steps >= 1500 || s->margin < 0 || s->warning || !finite;
    s->success = s->steps == 1500 && s->best >= 250 && s->margin >= 0 && !s->warning && finite;
    return 1;
}

static void sp_observe(const SpatialCore* s, float* obs) {
    for (int j = 0; j < SP_NQ; ++j) obs[j] = (float)(s->state[j] / (j < 2 ? 3. : 1.));
    for (int j = 0; j < SP_NV; ++j) obs[SP_NQ + j] = (float)(s->state[SP_NQ + j] / (j < 2 ? 4. : 15.));
    obs[SP_STATE] = s->delivered[0] / 80.f;
    obs[SP_STATE + 1] = s->delivered[1] / 80.f;
    obs[SP_STATE + 2] = (float)(s->steps / 1500.);
}

static double sp_reward(const SpatialCore* s, int normalized_effort) {
    double upright = 0;
    for (int i = 0; i < SPATIAL_LINKS; ++i) upright += cos(s->tilts[i]);
    double force2 = (double)s->delivered[0] * s->delivered[0] + (double)s->delivered[1] * s->delivered[1];
    double penalty = normalized_effort ? .01 / (80. * 80.) : .001;
    return upright / SPATIAL_LINKS - .03 * (s->state[0] * s->state[0] + s->state[1] * s->state[1]) - penalty * force2;
}

static void sp_close(SpatialCore* s) { if (s->data) mj_deleteData(s->data); s->data = NULL; }
