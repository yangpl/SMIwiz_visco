/* Tapenade-facing acoustic modelling loop. */
/* TAPENADE-IGNORE-BEGIN */
#include <mpi.h>
/* TAPENADE-IGNORE-END */
#include "cstd.h"
#include "sim.h"
#include "acq.h"


void fdtd_update_v(sim_t *sim, float ***p, float ***vz, float ***vx,
                   float ***vy, float ***memD1p, float ***memD2p,
                   float ***memD3p, float ***buz, float ***bux,
                   float ***buy);
void fdtd_update_p(sim_t *sim, float ***p, float ***vz, float ***vx,
                   float ***vy, float ***memD1vz, float ***memD2vx,
                   float ***memD3vy, float ***xi1, float ***xi2,
                   float ***xi3, float ***kappa, float ***qinvmod);
void inject_source(sim_t *sim, acq_t *acq, float ***p, float stf_it);
void extract_wavefield(sim_t *sim, acq_t *acq, float ***p, float **dcal,
                       int it);
void convert_parameters(sim_t *sim, float ***vp, float ***rho,
                        float ***qinv,
                        float ***vpmod, float ***rhomod, float ***qinvmod,
                        float ***kappa, float ***buz, float ***bux,
                        float ***buy);

void modelling(sim_t *sim, acq_t *acq,
               float ***p, float ***vz, float ***vx, float ***vy,
               float ***memD1p, float ***memD2p, float ***memD3p,
               float ***memD1vz, float ***memD2vx, float ***memD3vy,
               float ***xi1, float ***xi2, float ***xi3,
               float ***vp, float ***rho, float ***qinv,
               float ***vpmod, float ***rhomod, float ***qinvmod,
               float ***kappa, float ***buz, float ***bux, float ***buy,
               float **dcal)
{
  int it;
  int nt = sim->nt;
  int nsnap = sim->nsnap;
  /* TAPENADE-IGNORE-BEGIN */
  double t0, t_update_v, t_update_p, t_inject_src, t_extract_field;
  
  t_update_v = 0.;
  t_update_p = 0.;
  t_inject_src = 0.;
  t_extract_field = 0.;
  /* TAPENADE-IGNORE-END */
  (void)nsnap; /* Referenced by Tapenade's BINOMIAL-CKP directive. */
  convert_parameters(sim, vp, rho, qinv, vpmod, rhomod, qinvmod,
                     kappa, buz, bux, buy);
  
  /* $AD BINOMIAL-CKP nt nsnap 0 */
  for(it=0; it<nt; it++){
    if(iproc==0 && it%100==0) printf("it-----%d\n", it);

    /* TAPENADE-IGNORE-BEGIN */
    if(iproc==0) t0 = MPI_Wtime();
    /* TAPENADE-IGNORE-END */
    fdtd_update_v(sim, p, vz, vx, vy, memD1p, memD2p, memD3p,
                  buz, bux, buy);
    /* TAPENADE-IGNORE-BEGIN */
    if(iproc==0) t_update_v += MPI_Wtime()-t0;
    /* TAPENADE-IGNORE-END */
    
    /* TAPENADE-IGNORE-BEGIN */
    if(iproc==0) t0 = MPI_Wtime();
    /* TAPENADE-IGNORE-END */
    fdtd_update_p(sim, p, vz, vx, vy, memD1vz, memD2vx, memD3vy,
                  xi1, xi2, xi3, kappa, qinvmod);
    /* TAPENADE-IGNORE-BEGIN */
    if(iproc==0) t_update_p += MPI_Wtime()-t0;
    /* TAPENADE-IGNORE-END */

    /* TAPENADE-IGNORE-BEGIN */
    if(iproc==0) t0 = MPI_Wtime();
    /* TAPENADE-IGNORE-END */
    inject_source(sim, acq, p, sim->stf[it]);
    /* TAPENADE-IGNORE-BEGIN */
    if(iproc==0) t_inject_src += MPI_Wtime()-t0;
    /* TAPENADE-IGNORE-END */

    /* TAPENADE-IGNORE-BEGIN */
    if(iproc==0) t0 = MPI_Wtime();
    /* TAPENADE-IGNORE-END */
    extract_wavefield(sim, acq, p, dcal, it);
    /* TAPENADE-IGNORE-BEGIN */
    if(iproc==0) t_extract_field += MPI_Wtime()-t0;
    /* TAPENADE-IGNORE-END */
    
  }
  
  /* TAPENADE-IGNORE-BEGIN */
  if(iproc==0 && sim->mode==0) {
    t0 = t_update_v + t_update_p + t_inject_src + t_extract_field;
    FILE *fp = fopen("time_info.txt", "w");
    if(fp==NULL) err("cannot open time_info.txt for writing");
    fprintf(fp, "update_v      \t %e\n", t_update_v);
    fprintf(fp, "update_p      \t %e\n", t_update_p);
    fprintf(fp, "inject_src    \t %e\n", t_inject_src);
    fprintf(fp, "extract_field \t %e\n", t_extract_field);
    fclose(fp);
    
    printf("-------------- elapsed time --------------------\n");
    printf("update_v      \t %e\n", t_update_v);
    printf("update_p      \t %e\n", t_update_p);
    printf("inject_src    \t %e\n", t_inject_src);
    printf("extract_field \t %e\n", t_extract_field);
    printf("total time    \t %e\n", t0);
    printf("------------------------------------------------\n");
  }
  /* TAPENADE-IGNORE-END */

}
