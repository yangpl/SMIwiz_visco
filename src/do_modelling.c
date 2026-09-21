/* 2D/3D isotropic acoustic forward modelling
 *-----------------------------------------------------------------------
 *  Copyright (c) Pengliang Yang, 2026, Laoshan Laboratory, China
 *  Copyright (c) Pengliang Yang, 2020, Harbin Institute of Technology, China
 *  Copyright (c) Pengliang Yang, 2018, University Grenoble Alpes, France
 *  Homepage: https://yangpl.wordpress.com
 *  E-mail: ypl.2100@gmail.com
 *--------------------------------------------------------------------------*/
#include "cstd.h"
#include "sim.h"
#include "acq.h"

void check_cfl(sim_t *sim);

void fdtd_init(sim_t *sim, int adj);
void fdtd_null(sim_t *sim, int adj);
void fdtd_free(sim_t *sim, int adj);

void extend_model_init(sim_t *sim);
void extend_model_free(sim_t *sim);

void cpml_init(sim_t *sim);
void cpml_free(sim_t *sim);


void modelling(sim_t *sim, acq_t *acq,
               float ***p, float ***vz, float ***vx, float ***vy,
               float ***memD1p, float ***memD2p, float ***memD3p,
               float ***memD1vz, float ***memD2vx, float ***memD3vy,
               float ***xi1, float ***xi2, float ***xi3,
               float ***vp, float ***rho, float ***qinv,
               float ***vpmod, float ***rhomod, float ***qinvmod,
               float ***kappa, float ***buz, float ***bux, float ***buy,
               float **dcal);

void write_data(sim_t *sim, acq_t *acq);

void do_modelling(sim_t *sim, acq_t *acq)
{
  check_cfl(sim);

  cpml_init(sim);
  extend_model_init(sim);
  fdtd_init(sim, 0);
  fdtd_null(sim, 0);

  modelling(sim,acq,sim->p,sim->vz,sim->vx,sim->vy,
            sim->memD1p,sim->memD2p,sim->memD3p,
            sim->memD1vz,sim->memD2vx,sim->memD3vy,
            sim->xi1,sim->xi2,sim->xi3,
            sim->vp,sim->rho,sim->qinv,
            sim->vpmod,sim->rhomod,sim->qinvmod,
            sim->kappa,sim->buz,sim->bux,sim->buy,
            sim->dcal);
  write_data(sim, acq);
  
  extend_model_free(sim);
  fdtd_free(sim, 0);
  cpml_free(sim);

}
