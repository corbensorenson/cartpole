/* Test/benchmark exports, independent of the PufferLib CUDA trainer. */
#include <vector>
#include <new>
#include "spatial_cartpole.h"

#ifdef _WIN32
#define SP_EXPORT extern "C" __declspec(dllexport)
#else
#define SP_EXPORT extern "C" __attribute__((visibility("default")))
#endif

struct NativeBatch {
    mjModel* model;
    std::vector<SpatialCore> envs;
    std::vector<float> observations;
    std::vector<double> rewards;
};

static void metrics(const SpatialCore* s, double* m) {
    m[0] = s->steps; m[1] = s->streak; m[2] = s->best;
    m[3] = s->margin; m[4] = s->warning; m[5] = s->done;
    m[6] = s->success; m[7] = (double)s->native_steps; m[8] = s->max_tilt;
}

SP_EXPORT int ns_state_size() { return SP_STATE; }
SP_EXPORT int ns_obs_size() { return SP_OBS; }
SP_EXPORT void* ns_create(const char* xml, int count) {
    if (count < 1) return NULL;
    NativeBatch* b = new NativeBatch;
    char error[1024];
    b->model = mj_loadXML(xml, NULL, error, sizeof(error));
    if (!b->model) { fprintf(stderr, "%s\n", error); delete b; return NULL; }
    b->envs.resize(count);
    b->observations.resize(count * SP_OBS);
    b->rewards.resize(count);
    for (int i = 0; i < count; ++i) {
        if (!sp_init(&b->envs[i], b->model)) {
            for (int j = 0; j < i; ++j) sp_close(&b->envs[j]);
            mj_deleteModel(b->model); delete b; return NULL;
        }
    }
    return b;
}
SP_EXPORT void ns_set(void* ptr, int i, const double* state) {
    NativeBatch* b = (NativeBatch*)ptr;
    sp_reset_state(&b->envs.at(i), state);
}
SP_EXPORT void ns_step(void* ptr, const double* forces, int threads) {
    NativeBatch* b = (NativeBatch*)ptr;
    int count = (int)b->envs.size();
    #pragma omp parallel for num_threads(threads) if(threads > 1)
    for (int i = 0; i < count; ++i) {
        sp_step(&b->envs[i], forces + 2 * i);
        sp_observe(&b->envs[i], b->observations.data() + i * SP_OBS);
        b->rewards[i] = sp_reward(&b->envs[i], 0);
    }
}
SP_EXPORT void ns_read(void* ptr, int i, double* state, double* m, float* action, double* tilts) {
    NativeBatch* b = (NativeBatch*)ptr;
    const SpatialCore* s = &b->envs.at(i);
    memcpy(state, s->state, SP_STATE * sizeof(double));
    memcpy(action, s->delivered, 2 * sizeof(float));
    memcpy(tilts, s->tilts, SPATIAL_LINKS * sizeof(double));
    metrics(s, m);
}
SP_EXPORT uint64_t ns_cost(void* ptr) {
    uint64_t total = 0;
    for (const SpatialCore& s : ((NativeBatch*)ptr)->envs) total += s.native_steps;
    return total;
}
SP_EXPORT void ns_close(void* ptr) {
    NativeBatch* b = (NativeBatch*)ptr;
    for (SpatialCore& s : b->envs) sp_close(&s);
    mj_deleteModel(b->model); delete b;
}

struct PufferSmoke {
    Env env;
    float obs[SP_OBS], action[2], reward[1], terminal[1];
};
SP_EXPORT void* np_create(const char* xml, unsigned int seed, double noise) {
    PufferSmoke* b = new PufferSmoke();
    b->env.rng = seed;
    Dict config = {0};
    dict_set_str(&config, "model_path", xml);
    dict_set(&config, "angle_noise", noise);
    dict_set(&config, "normalized_effort", 1);
    puf_init(&b->env, &config);
    dict_clear(&config);
    b->env.agents[0].observations = b->obs;
    b->env.agents[0].actions = b->action;
    b->env.agents[0].rewards = b->reward;
    b->env.agents[0].terminals = b->terminal;
    puf_reset(&b->env);
    return b;
}
SP_EXPORT void np_set(void* ptr, const double* state) {
    PufferSmoke* b = (PufferSmoke*)ptr;
    sp_reset_state(&b->env.core, state);
    b->env.episode_return = 0;
    sp_observe(&b->env.core, b->obs);
}
SP_EXPORT void np_step(void* ptr, const float* action) {
    PufferSmoke* b = (PufferSmoke*)ptr;
    memcpy(b->action, action, 2 * sizeof(float)); puf_step(&b->env);
}
SP_EXPORT void np_read(void* ptr, double* state, double* m, float* obs, float* event) {
    PufferSmoke* b = (PufferSmoke*)ptr;
    memcpy(state, b->env.core.state, SP_STATE * sizeof(double));
    metrics(&b->env.core, m);
    memcpy(obs, b->obs, SP_OBS * sizeof(float));
    event[0] = b->reward[0]; event[1] = b->terminal[0];
    event[2] = b->env.log.n; event[3] = b->env.log.perf;
    event[4] = b->env.log.score; event[5] = (float)b->env.boundary_reached;
}
SP_EXPORT void np_close(void* ptr) {
    PufferSmoke* b = (PufferSmoke*)ptr;
    puf_close(&b->env); delete b;
}

#ifdef SP_EVALUATION_IO
SP_EXPORT void* np_create_eval(const char* xml, const char* plan, const char* directory) {
    PufferSmoke* b = new PufferSmoke();
    Dict config = {0};
    dict_set_str(&config, "model_path", xml);
    dict_set(&config, "angle_noise", .02);
    dict_set(&config, "normalized_effort", 1);
    dict_set_str(&config, "evaluation_plan", plan);
    dict_set_str(&config, "trace_directory", directory);
    char accounting[8192];
    snprintf(accounting, sizeof(accounting), "%s/accounting.json", directory);
    dict_set_str(&config, "accounting_file", accounting);
    puf_init(&b->env, &config);
    dict_clear(&config);
    b->env.agents[0].observations = b->obs;
    b->env.agents[0].actions = b->action;
    b->env.agents[0].rewards = b->reward;
    b->env.agents[0].terminals = b->terminal;
    puf_reset(&b->env);
    return b;
}
#endif
