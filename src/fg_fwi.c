/* Function and Tapenade-generated gradient evaluation for acoustic FWI. */
#include <mpi.h>
#include "cstd.h"
#include "sim.h"
#include "acq.h"
#include "fwi.h"
#include "modelling.h"

extern sim_t *sim;
extern acq_t *acq;
extern fwi_t *fwi;

void check_cfl(sim_t *sim);

void fdtd_init(sim_t *sim, int adj);
void fdtd_null(sim_t *sim, int adj);
void fdtd_free(sim_t *sim, int adj);

void extend_model_init(sim_t *sim);
void extend_model_free(sim_t *sim);

void cpml_init(sim_t *sim);
void cpml_free(sim_t *sim);

/* Keep the physical fields synchronized with the logarithmic optimizer state
 * before either a gradient or Gauss-Newton Hessian-vector evaluation. */
static void fwi_set_model(float *x)
{
  int i1, i2, i3, ipar, j;
  float value;

  for(ipar=0; ipar<fwi->npar; ipar++){
    for(i3=0; i3<sim->n3; i3++){
      for(i2=0; i2<sim->n2; i2++){
        for(i1=0; i1<sim->n1; i1++){
          j = i1 + sim->n1*(i2 + sim->n2*(i3 + sim->n3*ipar));
          if(fwi->idxpar[ipar]==1)
            sim->vp[i3][i2][i1] = expf(x[j]);
          if(fwi->idxpar[ipar]==3)
            sim->qinv[i3][i2][i1] = expf(x[j]);
        }
      }
    }
  }
  for(ipar=0; ipar<fwi->npar; ipar++){
    if(fwi->idxpar[ipar]!=2) continue;
    for(i3=0; i3<sim->n3; i3++){
      for(i2=0; i2<sim->n2; i2++){
        for(i1=0; i1<sim->n1; i1++){
          j = i1 + sim->n1*(i2 + sim->n2*(i3 + sim->n3*ipar));
            value = expf(x[j]);
            sim->rho[i3][i2][i1] = fwi->family==1 ? value
                                                   : value/sim->vp[i3][i2][i1];
        }
      }
    }
  }
}

static void fwi_apply_bathymetry_mask(float *g)
{
  int i1, i2, i3, ipar, j;

  for(ipar=0; ipar<fwi->npar; ipar++)
    for(i3=0; i3<sim->n3; i3++)
      for(i2=0; i2<sim->n2; i2++)
        for(i1=0; i1<=fwi->ibathy[i3][i2]; i1++){
          j = i1 + sim->n1*(i2 + sim->n2*(i3 + sim->n3*ipar));
          g[j] = 0.0f;
        }
}

double fg_fwi_eval(float *x, float *g)
{
  int i1,i2,i3,ipar,j,it,irec;
  double local_cost, tmp;
  float s1, s2;
  float dJ_dvp, dJ_drho, dJ_dqinv;
  float ***kappab, ***buzb, ***buxb, ***buyb;
  float ***vpb, ***rhob, ***qinvb, ***vpmodb, ***rhomodb, ***qinvmodb;
  float **dcalb;
  float *greduced;
  char fname[sizeof("dsyn_0000")];
  FILE *fp;

  memset(g, 0, fwi->n*sizeof(float));
  fwi_set_model(x);
  kappab = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
  buzb = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
  buxb = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
  buyb = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
  vpb = alloc3float(sim->n1, sim->n2, sim->n3);
  rhob = alloc3float(sim->n1, sim->n2, sim->n3);
  qinvb = alloc3float(sim->n1, sim->n2, sim->n3);
  vpmodb = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
  rhomodb = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
  qinvmodb = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
  dcalb = alloc2float(sim->nt, acq->nrec);

  check_cfl(sim);
  cpml_init(sim);
  extend_model_init(sim);
  fdtd_init(sim, 0);//adj=0
  fdtd_init(sim, 1);//adj=1
  fdtd_null(sim, 0);//adj=0
  fdtd_null(sim, 1);//adj=1

  if(iproc==0) printf("----stage 1: Tapenade primal modelling!--------\n");
  modelling(sim,acq,sim->p,sim->vz,sim->vx,sim->vy,
            sim->memD1p,sim->memD2p,sim->memD3p,
            sim->memD1vz,sim->memD2vx,sim->memD3vy,
            sim->xi1,sim->xi2,sim->xi3,
            sim->vp,sim->rho,sim->qinv,
            sim->vpmod,sim->rhomod,sim->qinvmod,
            sim->kappa,sim->buz,sim->bux,sim->buy,
            sim->dcal);
  local_cost = 0;
  for(irec=0; irec<acq->nrec; irec++) for(it=0; it<sim->nt; it++){
    const double w = (double)acq->wdat[irec][it];
    const double dres = (double)sim->dcal[irec][it]
                      - (double)sim->dobs[irec][it];
    /* Keep wavefields and data in float, but accumulate the objective and
     * form its data-space adjoint seed in double precision. */
    tmp = w*dres;
    local_cost += 0.5*tmp*tmp;
    dcalb[irec][it] = (float)(w*w*dres);
  }
  MPI_Allreduce(&local_cost,&tmp,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  fwi->fcost = tmp;
  if(iproc==0) printf("fcost=%e\n",fwi->fcost);

  sprintf(fname,"dsyn_%04d",acq->shot_idx[iproc]);
  fp=fopen(fname,"wb");
  if(fp==NULL) err("cannot open synthetic data file=%s",fname);
  if(fwrite(sim->dcal[0],sizeof(float),(size_t)sim->nt*acq->nrec,fp)
     !=(size_t)sim->nt*acq->nrec) err("cannot write %s",fname);
  if(fclose(fp)!=0) err("cannot close %s",fname);

  
  if(iproc==0) printf("----stage 2: Tapenade reverse modelling!-------\n");
  memset(kappab[0][0],0,sim->n123pad*sizeof(float));
  memset(buzb[0][0],0,sim->n123pad*sizeof(float));
  memset(buxb[0][0],0,sim->n123pad*sizeof(float));
  memset(buyb[0][0],0,sim->n123pad*sizeof(float));
  memset(vpb[0][0],0,sim->n123*sizeof(float));
  memset(rhob[0][0],0,sim->n123*sizeof(float));
  memset(qinvb[0][0],0,sim->n123*sizeof(float));
  memset(vpmodb[0][0],0,sim->n123pad*sizeof(float));
  memset(rhomodb[0][0],0,sim->n123pad*sizeof(float));
  memset(qinvmodb[0][0],0,sim->n123pad*sizeof(float));
  fdtd_null(sim, 0);//adj=0
  fdtd_null(sim, 1);//adj=1
  modelling_b(sim,acq,sim->p,sim->pb,sim->vz,sim->vzb,
              sim->vx,sim->vxb,sim->vy,sim->vyb,
              sim->memD1p,sim->memD1pb,
              sim->memD2p,sim->memD2pb,
              sim->memD3p,sim->memD3pb,
              sim->memD1vz,sim->memD1vzb,
              sim->memD2vx,sim->memD2vxb,
              sim->memD3vy,sim->memD3vyb,
              sim->xi1,sim->xi1b,
              sim->xi2,sim->xi2b,
              sim->xi3,sim->xi3b,
              sim->vp,vpb,sim->rho,rhob,
              sim->qinv,qinvb,
              sim->vpmod,vpmodb,sim->rhomod,rhomodb,
              sim->qinvmod,qinvmodb,
              sim->kappa,kappab,sim->buz,buzb,sim->bux,buxb,sim->buy,buyb,
              sim->dcal,dcalb);

  for(ipar=0; ipar<fwi->npar; ipar++){
    for(i3=0; i3<sim->n3; i3++){
      for(i2=0; i2<sim->n2; i2++){
        for(i1=0; i1<sim->n1; i1++){
          j = i1 + sim->n1*(i2 + sim->n2*(i3 + sim->n3*ipar));
          /* Tapenade returns derivatives with respect to the physical vp,
           * rho, and qinv fields. Apply the chain rule only once for the
           * logarithmic inversion parameters. */
          dJ_dvp = vpb[i3][i2][i1];
          dJ_drho = rhob[i3][i2][i1];
          dJ_dqinv = qinvb[i3][i2][i1];
          if(fwi->idxpar[ipar]==3)
            g[j] = sim->qinv[i3][i2][i1]*dJ_dqinv;
          if(fwi->family==1){
            /* family 1: (m1,m2)=(ln(vp),ln(rho)). */
            if(fwi->idxpar[ipar]==1)
              g[j] = sim->vp[i3][i2][i1]*dJ_dvp;
            if(fwi->idxpar[ipar]==2)
              g[j] = sim->rho[i3][i2][i1]*dJ_drho;
          }else{
            /* family 2: (m1,m2)=(ln(vp),ln(Ip)), Ip=rho*vp. */
            if(fwi->idxpar[ipar]==1)
              g[j] = sim->vp[i3][i2][i1]*dJ_dvp
                   - sim->rho[i3][i2][i1]*dJ_drho;
            if(fwi->idxpar[ipar]==2)
              g[j] = sim->rho[i3][i2][i1]*dJ_drho;
          }
        }//end for i1
      }//end for i2
    }//end for i3
  }//end for ipar

  greduced = alloc1float(sim->n123);
  for(ipar=0; ipar<fwi->npar; ipar++){
    MPI_Allreduce(&g[ipar*sim->n123],greduced,sim->n123, MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
    memcpy(&g[ipar*sim->n123],greduced,sim->n123*sizeof(float));
  }
  free1float(greduced);

  free2float(dcalb);
  free3float(kappab);
  free3float(buzb);
  free3float(buxb);
  free3float(buyb);
  free3float(vpb);
  free3float(rhob);
  free3float(qinvb);
  free3float(vpmodb);
  free3float(rhomodb);
  free3float(qinvmodb);
  fdtd_free(sim, 0);//adj=0
  fdtd_free(sim, 1);//adj=1
  extend_model_free(sim);
  cpml_free(sim);

  fwi_apply_bathymetry_mask(g);
  if(iproc==0 && fwi->firstgrad){
    const char *gradient_file = sim->mode==2 ? "image_rtm" : "gradient_fwi";
    fp=fopen(gradient_file,"wb");
    if(fp==NULL) err("cannot open %s for writing", gradient_file);
    if(fwrite(g,sizeof(float),fwi->n,fp)!=(size_t)fwi->n)
      err("cannot write %s", gradient_file);
    if(fclose(fp)!=0) err("cannot close %s", gradient_file);
  }

  if(sim->mode==1 && fwi->firstgrad){
    s1 = fabsf(x[0]);
    s2 = fabsf(g[0]);
    for(j=0; j<fwi->n; j++){
      s1 = MAX(s1,fabsf(x[j]));
      s2 = MAX(s2,fabsf(g[j]));
    }
    fwi->alpha = s2>0.0f ? 0.01f*s1/s2 : 1.0f;
    fwi->firstgrad = 0;
    if(iproc==0) printf("scaling=%e\n",fwi->alpha);
  }
  if(sim->mode!=4){
    if(!fwi->firstgrad)
      for(j=0; j<fwi->n; j++) g[j] *= fwi->alpha;
    fwi->fcost *= fwi->alpha;
  }
  if(iproc==0) printf("scaled fcost=%g\n",fwi->fcost);
  return fwi->fcost;
}

/* The optimizer still consumes the historical float callback.  Keep the
 * double-precision objective available to validation and cast only at this
 * legacy interface boundary. */
float fg_fwi(float *x, float *g)
{
  return (float)fg_fwi_eval(x, g);
}

void Hv_fwi(float *x, float *v, float *Hv)
/*< apply the scaled data Gauss-Newton Hessian at the current FWI model >*/
{
  int i1, i2, i3, ipar, j;

  fwi_set_model(x);
  lsm_hessian_vector(sim, acq, fwi->family, fwi->npar, fwi->idxpar, v, Hv);
  for(j=0; j<fwi->n; j++) Hv[j] *= fwi->alpha;

  /* Match the bathymetry projection applied to the FWI gradient. */
  for(ipar=0; ipar<fwi->npar; ipar++){
    for(i3=0; i3<sim->n3; i3++){
      for(i2=0; i2<sim->n2; i2++){
        for(i1=0; i1<=fwi->ibathy[i3][i2]; i1++){
          j = i1 + sim->n1*(i2 + sim->n2*(i3 + sim->n3*ipar));
          Hv[j] = 0.0f;
        }
      }
    }
  }
}
