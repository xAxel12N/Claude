/* gpu.h - train QuantumGPT on a graphics card through OpenCL (NVIDIA, AMD or Intel).
 * The card holds 32-bit copies of the weights and optimiser state; the CPU Model stays
 * the master copy for sampling, the brain view and checkpoints (sync with gpu_pull). */
#ifndef GPU_H
#define GPU_H

#include "qcore.h"

typedef struct Gpu Gpu;

/* Finds an OpenCL GPU. Returns NULL and a reason in err when there is none.
 * QGPT_OPENCL=any also accepts CPU OpenCL devices (for testing). */
Gpu *gpu_open(char *err, size_t errn);
const char *gpu_name(const Gpu *g);
void gpu_close(Gpu *g);

/* (Re)allocates the card's buffers for this model and batch size and uploads the weights
 * (and the optimiser state, if s is given). */
int gpu_bind(Gpu *g, const Model *m, const OptState *s, int B);
/* One training step on the card. Returns the mean loss, NAN on failure. */
double gpu_step(Gpu *g, const Model *m, OptState *s, const int *X, const int *Y, const OptCfg *o, double lr,
                double *gnorm, uint64_t *rng);
/* Mean loss over B sequences, no learning. */
double gpu_eval(Gpu *g, const Model *m, const int *X, const int *Y);
/* Copies the weights (and the optimiser state, if s is given) back into the CPU model. */
int gpu_pull(Gpu *g, Model *m, OptState *s);
/* Testing: loss and gradient for one batch, as doubles in the CPU layout. */
int gpu_grad(Gpu *g, const Model *m, const int *X, const int *Y, double *grad, double *loss);
const char *gpu_error(const Gpu *g);

#endif
