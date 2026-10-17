#define SPATIAL_LINKS 1
#pragma once
/* PufferLib 5.0 ea7f989 native CPU environment / CUDA policy interface. */
typedef float obs_t;
#include "pufferenv.h"
#include "spatial_core.h"
#ifdef SP_EVALUATION_IO
#include "evaluation_io.h"
#endif

#define OBS_SIZE SP_OBS
#define NUM_ATNS 2
#define ACT_SIZES {1, 1}

struct Log {
    float perf, score, episode_return, episode_length, hold_seconds;
    float rail_exits, physics_warnings, n;
};

struct Env {
    Log log;
    Agent agents[1];
    int num_agents, tag, boundary_reached;
    unsigned int rng;
    uint64_t reset_rng;
    SpatialCore core;
    double angle_noise, episode_return;
    int normalized_effort;
#ifdef SP_EVALUATION_IO
    SpatialEval evaluation;
#endif
};

/* PufferLib initializes and closes CPU environments serially. */
static mjModel* spatial_shared_model = NULL;
static int spatial_model_users = 0;
static uint64_t spatial_closed_native_steps = 0;
static unsigned int spatial_closed_envs = 0;
static char spatial_accounting_file[4096];

void puf_init(Env* env, Dict* kwargs) {
    if (!spatial_shared_model) {
        spatial_closed_native_steps = 0;
        spatial_closed_envs = 0;
        spatial_accounting_file[0] = 0;
        DictItem* accounting = dict_find(kwargs, "accounting_file");
        if (accounting && accounting->str && strcmp(accounting->str, "None")) {
            if (strlen(accounting->str) >= sizeof(spatial_accounting_file)) {
                fprintf(stderr, "spatial accounting path too long\n"); exit(1);
            }
            strcpy(spatial_accounting_file, accounting->str);
        }
        char error[1024];
        spatial_shared_model = mj_loadXML(dict_get_str(kwargs, "model_path"), NULL, error, sizeof(error));
        if (!spatial_shared_model) { fprintf(stderr, "spatial model: %s\n", error); exit(1); }
    }
    if (!sp_init(&env->core, spatial_shared_model)) { fprintf(stderr, "spatial model dimensions/options mismatch\n"); exit(1); }
    spatial_model_users++;
    env->num_agents = 1;
    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;
    env->reset_rng = (uint64_t)env->rng + 1;
    env->angle_noise = dict_get(kwargs, "angle_noise");
    env->normalized_effort = (int)dict_get(kwargs, "normalized_effort");
    if (env->angle_noise < 0 || !isfinite(env->angle_noise)) { fprintf(stderr, "invalid reset noise\n"); exit(1); }
#ifdef SP_EVALUATION_IO
    DictItem* plan = dict_find(kwargs, "evaluation_plan");
    if (plan && plan->str && strcmp(plan->str, "None")) {
        /* Evaluation uses exactly one agent; seed identity never enters observations. */
        if (env->rng != 0) sp_io_fail("recorded evaluation requires a single environment");
        sp_eval_init(&env->evaluation, plan->str, dict_get_str(kwargs, "trace_directory"));
    }
#endif
}

void puf_reset(Env* env) {
#ifdef SP_EVALUATION_IO
    if (env->evaluation.plan) {
        if (!sp_eval_reset(&env->evaluation, &env->core)) {
            memset(env->agents[0].observations, 0, SP_OBS * sizeof(float));
            return;
        }
    } else
#endif
    sp_reset_random(&env->core, &env->reset_rng, env->angle_noise);
    env->episode_return = 0;
    env->boundary_reached = 0;
    sp_observe(&env->core, env->agents[0].observations);
}

void puf_step(Env* env) {
    Agent* a = &env->agents[0];
    a->rewards[0] = a->terminals[0] = 0;
#ifdef SP_EVALUATION_IO
    if (env->evaluation.exhausted) return;
#endif
    double force[2] = {(double)a->actions[0] * 80., (double)a->actions[1] * 80.};
    int advanced = sp_step(&env->core, force);
    double reward = advanced ? sp_reward(&env->core, env->normalized_effort) : -1.;
    a->rewards[0] = (float)reward;
#ifdef SP_EVALUATION_IO
    if (env->evaluation.plan)
        sp_eval_record(&env->evaluation, &env->core, a->observations, a->actions, advanced, a->rewards[0]);
#endif
    env->episode_return += reward;
    env->boundary_reached = env->core.done;
    if (env->core.done) {
        env->log.perf += env->core.success;
        env->log.score += env->core.best * .02f;
        env->log.episode_return += (float)env->episode_return;
        env->log.episode_length += (float)env->core.steps;
        env->log.hold_seconds += env->core.best * .02f;
        env->log.rail_exits += env->core.margin < 0;
        env->log.physics_warnings += env->core.warning;
        env->log.n++;
        a->terminals[0] = 1;
        puf_reset(env);
        env->boundary_reached = 1;
    } else sp_observe(&env->core, a->observations);
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "hold_seconds", log->hold_seconds);
    dict_set(out, "rail_exits", log->rail_exits);
    dict_set(out, "physics_warnings", log->physics_warnings);
    dict_set(out, "n", log->n);
}

/* Qualification videos use the original full-state MuJoCo tape renderer. */
void puf_render(Env* env) { (void)env; }

void puf_close(Env* env) {
#ifdef SP_EVALUATION_IO
    sp_eval_close(&env->evaluation);
#endif
    spatial_closed_native_steps += env->core.native_steps;
    spatial_closed_envs++;
    sp_close(&env->core);
    if (--spatial_model_users == 0) {
        mj_deleteModel(spatial_shared_model); spatial_shared_model = NULL;
        if (spatial_accounting_file[0]) {
            FILE* file = fopen(spatial_accounting_file, "wbx");
            if (!file) { fprintf(stderr, "spatial accounting exists or cannot be created\n"); exit(1); }
            int written = fprintf(file,
                "{\"schema\":1,\"links\":%d,\"closed_environments\":%u,\"native_steps\":%llu,\"control_transitions\":%llu}\n",
                SPATIAL_LINKS, spatial_closed_envs,
                (unsigned long long)spatial_closed_native_steps,
                (unsigned long long)(spatial_closed_native_steps / 4));
            if (fclose(file) || written < 0) { fprintf(stderr, "spatial accounting write failed\n"); exit(1); }
        }
    }
}
