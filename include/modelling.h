#ifndef _modelling_h_
#define _modelling_h_

#include "sim.h"
#include "acq.h"

void modelling(sim_t *sim, acq_t *acq,
               float ***p, float ***vz, float ***vx, float ***vy,
               float ***memD1p, float ***memD2p, float ***memD3p,
               float ***memD1vz, float ***memD2vx, float ***memD3vy,
               float ***xi1, float ***xi2, float ***xi3,
               float ***vp, float ***rho, float ***qinv,
               float ***vpmod, float ***rhomod, float ***qinvmod,
               float ***kappa, float ***buz, float ***bux, float ***buy,
               float **dcal);

/* Reverse of the calculated-data output. dcalb is an input seed, allowing
 * callers to apply J^T directly to an arbitrary data-space vector. */
void modelling_b(sim_t *sim, acq_t *acq,
                 float ***p, float ***pb, float ***vz, float ***vzb,
                 float ***vx, float ***vxb, float ***vy, float ***vyb,
                 float ***memD1p, float ***memD1pb,
                 float ***memD2p, float ***memD2pb,
                 float ***memD3p, float ***memD3pb,
                 float ***memD1vz, float ***memD1vzb,
                 float ***memD2vx, float ***memD2vxb,
                 float ***memD3vy, float ***memD3vyb,
                 float ***xi1, float ***xi1b,
                 float ***xi2, float ***xi2b,
                 float ***xi3, float ***xi3b,
                 float ***vp, float ***vpb,
                 float ***rho, float ***rhob, float ***qinv, float ***qinvb,
                 float ***vpmod, float ***vpmodb,
                 float ***rhomod, float ***rhomodb, float ***qinvmod,
                 float ***qinvmodb,
                 float ***kappa, float ***kappab,
                 float ***buz, float ***buzb,
                 float ***bux, float ***buxb,
                 float ***buy, float ***buyb,
                 float **dcal, float **dcalb);

/* Tapenade forward/tangent derivative of modelling().  The un-suffixed
 * arguments are the primal state and model; each matching *d argument is its
 * directional variation. The routine returns the primal calculated data and
 * its directional variation. */
void modelling_d(sim_t *sim, acq_t *acq,
                 float ***p, float ***pd, float ***vz, float ***vzd,
                 float ***vx, float ***vxd, float ***vy, float ***vyd,
                 float ***memD1p, float ***memD1pd,
                 float ***memD2p, float ***memD2pd,
                 float ***memD3p, float ***memD3pd,
                 float ***memD1vz, float ***memD1vzd,
                 float ***memD2vx, float ***memD2vxd,
                 float ***memD3vy, float ***memD3vyd,
                 float ***xi1, float ***xi1d,
                 float ***xi2, float ***xi2d,
                 float ***xi3, float ***xi3d,
                 float ***vp, float ***vpd,
                 float ***rho, float ***rhod,
                 float ***qinv, float ***qinvd,
                 float ***vpmod, float ***vpmodd,
                 float ***rhomod, float ***rhomodd,
                 float ***qinvmod, float ***qinvmodd,
                 float ***kappa, float ***kappad,
                 float ***buz, float ***buzd,
                 float ***bux, float ***buxd,
                  float ***buy, float ***buyd,
                  float **dcal, float **dcald);

/* Apply the weighted Gauss-Newton normal operator J^T W^T W J to a
 * logarithmic model-space direction. */
void lsm_hessian_vector(sim_t *sim, acq_t *acq, int family, int npar,
                        const int *idxpar, const float *v, float *Hv);

#endif
