/* Finite-difference time-domain (FDTD) modelling kernel
 *----------------------------------------------------------------------
 *  Copyright (c) Pengliang Yang, 2026, Laoshan Laboratory, China
 *  Copyright (c) Pengliang Yang, 2020, Harbin Institute of Technology, China
 *  Copyright (c) Pengliang Yang, 2018, University Grenoble Alpes, France
 *  Homepage: https://yangpl.wordpress.com
 *  E-mail: ypl.2100@gmail.com
 *--------------------------------------------------------------------------*/
#include "cstd.h"
#include "sim.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void fdtd_init(sim_t *sim, int adj)
{
  if(!adj){//forward field
    sim->p  = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
    sim->vz = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
    sim->vx = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
    sim->vy = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
    sim->memD1p = alloc3float(2*sim->nb, sim->n2pad, sim->n3pad);
    sim->memD2p = alloc3float(sim->n1pad, 2*sim->nb, sim->n3pad);
    sim->memD1vz = alloc3float(2*sim->nb, sim->n2pad, sim->n3pad);
    sim->memD2vx = alloc3float(sim->n1pad, 2*sim->nb, sim->n3pad);
    sim->xi1 = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
    sim->xi2 = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
    sim->xi3 = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
    if(sim->n3>1){
      sim->memD3p = alloc3float(sim->n1pad, sim->n2pad, 2*sim->nb);
      sim->memD3vy = alloc3float(sim->n1pad, sim->n2pad, 2*sim->nb);
    }else{
      sim->memD3p = alloc3float(1, 1, 1);
      sim->memD3vy = alloc3float(1, 1, 1);
    }
    
  }else{//adjoint field
    sim->pb  = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
    sim->vzb = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
    sim->vxb = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
    sim->vyb = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
    sim->memD1pb = alloc3float(2*sim->nb, sim->n2pad, sim->n3pad);
    sim->memD2pb = alloc3float(sim->n1pad, 2*sim->nb, sim->n3pad);
    sim->memD1vzb = alloc3float(2*sim->nb, sim->n2pad, sim->n3pad);
    sim->memD2vxb = alloc3float(sim->n1pad, 2*sim->nb, sim->n3pad);
    sim->xi1b = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
    sim->xi2b = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
    sim->xi3b = alloc3float(sim->n1pad, sim->n2pad, sim->n3pad);
    if(sim->n3>1){
      sim->memD3pb = alloc3float(sim->n1pad, sim->n2pad, 2*sim->nb);
      sim->memD3vyb = alloc3float(sim->n1pad, sim->n2pad, 2*sim->nb);
    }else{
      sim->memD3pb = alloc3float(1, 1, 1);
      sim->memD3vyb = alloc3float(1, 1, 1);
    }
    
  }
}

void fdtd_null(sim_t *sim, int adj)
{
  if(!adj){//forward field initialization
    memset(sim->p [0][0], 0, sim->n123pad*sizeof(float));
    memset(sim->vz[0][0], 0, sim->n123pad*sizeof(float));
    memset(sim->vx[0][0], 0, sim->n123pad*sizeof(float));
    memset(sim->vy[0][0], 0, sim->n123pad*sizeof(float));
    memset(sim->memD1p[0][0], 0, 2*sim->nb*sim->n2pad*sim->n3pad*sizeof(float));
    memset(sim->memD2p[0][0], 0, sim->n1pad*2*sim->nb*sim->n3pad*sizeof(float));
    memset(sim->memD1vz[0][0], 0, 2*sim->nb*sim->n2pad*sim->n3pad*sizeof(float));
    memset(sim->memD2vx[0][0], 0, sim->n1pad*2*sim->nb*sim->n3pad*sizeof(float));
    memset(sim->xi1[0][0], 0, sim->n123pad*sizeof(float));
    memset(sim->xi2[0][0], 0, sim->n123pad*sizeof(float));
    memset(sim->xi3[0][0], 0, sim->n123pad*sizeof(float));
    if(sim->n3>1){
      memset(sim->memD3p[0][0], 0, sim->n1pad*sim->n2pad*2*sim->nb*sizeof(float));
      memset(sim->memD3vy[0][0], 0, sim->n1pad*sim->n2pad*2*sim->nb*sizeof(float));
    }else{
      sim->memD3p[0][0][0] = 0.0f;
      sim->memD3vy[0][0][0] = 0.0f;
    }

  }else{//adjoint field initialization
    memset(sim->pb [0][0], 0, sim->n123pad*sizeof(float));
    memset(sim->vzb[0][0], 0, sim->n123pad*sizeof(float));
    memset(sim->vxb[0][0], 0, sim->n123pad*sizeof(float));
    memset(sim->vyb[0][0], 0, sim->n123pad*sizeof(float));
    memset(sim->memD1pb[0][0], 0, 2*sim->nb*sim->n2pad*sim->n3pad*sizeof(float));
    memset(sim->memD2pb[0][0], 0, sim->n1pad*2*sim->nb*sim->n3pad*sizeof(float));
    memset(sim->memD1vzb[0][0], 0, 2*sim->nb*sim->n2pad*sim->n3pad*sizeof(float));
    memset(sim->memD2vxb[0][0], 0, sim->n1pad*2*sim->nb*sim->n3pad*sizeof(float));
    memset(sim->xi1b[0][0], 0, sim->n123pad*sizeof(float));
    memset(sim->xi2b[0][0], 0, sim->n123pad*sizeof(float));
    memset(sim->xi3b[0][0], 0, sim->n123pad*sizeof(float));
    if(sim->n3>1){
      memset(sim->memD3pb[0][0], 0, sim->n1pad*sim->n2pad*2*sim->nb*sizeof(float));
      memset(sim->memD3vyb[0][0], 0, sim->n1pad*sim->n2pad*2*sim->nb*sizeof(float));
    }else{
      sim->memD3pb[0][0][0] = 0.0f;
      sim->memD3vyb[0][0][0] = 0.0f;
    }

  }
}


void fdtd_free(sim_t *sim, int adj)
{
  if(!adj){//free forward field
    free3float(sim->p);
    free3float(sim->vx);
    free3float(sim->vz);
    free3float(sim->memD1p);
    free3float(sim->memD2p);
    free3float(sim->memD1vz);
    free3float(sim->memD2vx);
    free3float(sim->xi1);
    free3float(sim->xi2);
    free3float(sim->xi3);
    free3float(sim->vy);
    free3float(sim->memD3p);
    free3float(sim->memD3vy);
  }else{//free adjoint field
    free3float(sim->pb);
    free3float(sim->vxb);
    free3float(sim->vzb);
    free3float(sim->memD1pb);
    free3float(sim->memD2pb);
    free3float(sim->memD1vzb);
    free3float(sim->memD2vxb);
    free3float(sim->xi1b);
    free3float(sim->xi2b);
    free3float(sim->xi3b);
    free3float(sim->vyb);
    free3float(sim->memD3pb);
    free3float(sim->memD3vyb);
  }
}



void fdtd_update_v(sim_t *sim, float ***p, float ***vz, float ***vx, float ***vy, float ***memD1p, float ***memD2p, float ***memD3p, float ***buz, float ***bux, float ***buy)
{
  int i1, i2, i3, j1, j2, j3, k1, k2, k3;
  float D1p, D2p, D3p;
  int i1min, i1max, i2min, i2max, i3min, i3max;
  float dt = sim->dt;
  float _d1 = 1./sim->d1;
  float _d2 = 1./sim->d2;
  float _d3 = 1./sim->d3;

  if(sim->order==4){
    i1min = 1;
    i1max = sim->n1pad-3;
    i2min = 1;
    i2max = sim->n2pad-3;
    i3min = (sim->n3>1)?1:0;
    i3max = (sim->n3>1)?(sim->n3pad-3):0;
  }else if(sim->order==8){
    i1min = 3;
    i1max = sim->n1pad-5;
    i2min = 3;
    i2max = sim->n2pad-5;
    i3min = (sim->n3>1)?3:0;
    i3max = (sim->n3>1)?(sim->n3pad-5):0;
  }
  if(sim->freesurf) i1min = sim->nb;

  
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3, D1p, D2p, D3p)		\
  shared(i1min, i1max, i2min, i2max, i3min, i3max, _d1, _d2, _d3, dt,	\
	 p, vz, vx, vy, memD1p, memD2p, memD3p, buz, bux, buy, sim)
#endif  
  for(i3=i3min; i3<=i3max; i3++){
    for(i2=i2min; i2<=i2max; i2++){
      for(i1=i1min; i1<=i1max; i1++){
	if(sim->order==4){
	  D1p = 1.125*(p[i3][i2][i1+1]-p[i3][i2][i1])
	    -0.041666666666666664*(p[i3][i2][i1+2]-p[i3][i2][i1-1]);
	  D2p = 1.125*(p[i3][i2+1][i1]-p[i3][i2][i1])
	    -0.041666666666666664*(p[i3][i2+2][i1]-p[i3][i2-1][i1]);
	}else if(sim->order==8){
	  D1p = 1.196289062500000*(p[i3][i2][i1+1]-p[i3][i2][i1])
	    -0.079752604166667*(p[i3][i2][i1+2]-p[i3][i2][i1-1])
	    +0.009570312500000*(p[i3][i2][i1+3]-p[i3][i2][i1-2])
	    -0.000697544642857*(p[i3][i2][i1+4]-p[i3][i2][i1-3]);
	  D2p = 1.196289062500000*(p[i3][i2+1][i1]-p[i3][i2][i1])
	    -0.079752604166667*(p[i3][i2+2][i1]-p[i3][i2-1][i1])
	    +0.009570312500000*(p[i3][i2+3][i1]-p[i3][i2-2][i1])
	    -0.000697544642857*(p[i3][i2+4][i1]-p[i3][i2-3][i1]);
	}	
	D1p *= _d1;
	D2p *= _d2;
      
	if(i1<sim->nb) {
	  memD1p[i3][i2][i1] = sim->pmlb_ph[i1]*memD1p[i3][i2][i1] + sim->pmla_ph[i1]*D1p;
	  D1p += memD1p[i3][i2][i1];
	}else if(i1>=sim->n1pad-sim->nb-1){
	  /* vz[i1] is located at i1+1/2.  Reflect it onto the left
	   * half-grid node n1pad-2-i1 so both PMLs use the same profile. */
	  j1 = sim->n1pad-2-i1;
	  k1 = j1 + sim->nb;
	  memD1p[i3][i2][k1] = sim->pmlb_ph[j1]*memD1p[i3][i2][k1] + sim->pmla_ph[j1]*D1p;
	  D1p += memD1p[i3][i2][k1];
	}
	if(i2<sim->nb) {
	  memD2p[i3][i2][i1] = sim->pmlb_ph[i2]*memD2p[i3][i2][i1] + sim->pmla_ph[i2]*D2p;
	  D2p += memD2p[i3][i2][i1];
	}else if(i2>=sim->n2pad-sim->nb-1){
	  j2 = sim->n2pad-2-i2;
	  k2 = j2 + sim->nb;
	  memD2p[i3][k2][i1] = sim->pmlb_ph[j2]*memD2p[i3][k2][i1] + sim->pmla_ph[j2]*D2p;
	  D2p += memD2p[i3][k2][i1];
	}

	vz[i3][i2][i1] -= dt*buz[i3][i2][i1]*D1p;
	vx[i3][i2][i1] -= dt*bux[i3][i2][i1]*D2p;
	
	if(sim->n3>1){
	  if(sim->order==4){
	    D3p = 1.125*(p[i3+1][i2][i1]-p[i3][i2][i1])
	      -0.041666666666666664*(p[i3+2][i2][i1]-p[i3-1][i2][i1]);
	  }else if(sim->order==8){
	    D3p = 1.196289062500000*(p[i3+1][i2][i1]-p[i3][i2][i1])
	      -0.079752604166667*(p[i3+2][i2][i1]-p[i3-1][i2][i1])
	      +0.009570312500000*(p[i3+3][i2][i1]-p[i3-2][i2][i1])
	      -0.000697544642857*(p[i3+4][i2][i1]-p[i3-3][i2][i1]);
	  }
	  D3p *= _d3;
	  if(i3<sim->nb){
	    memD3p[i3][i2][i1] = sim->pmlb_ph[i3]*memD3p[i3][i2][i1] + sim->pmla_ph[i3]*D3p;
	    D3p += memD3p[i3][i2][i1];
	  }else if(i3>=sim->n3pad-sim->nb-1){
	    j3 = sim->n3pad-2-i3;
	    k3 = j3 + sim->nb;
	    memD3p[k3][i2][i1] = sim->pmlb_ph[j3]*memD3p[k3][i2][i1] + sim->pmla_ph[j3]*D3p;
	    D3p += memD3p[k3][i2][i1];
	  }

	  vy[i3][i2][i1] -= dt*buy[i3][i2][i1]*D3p;
	}//end if n3>1
	
      }//end for i3
    }//end for i2
  }//end for i1
  
  if(sim->freesurf){
    if(sim->order==4){
#ifdef _OPENMP
#pragma omp parallel for default(none)		\
  schedule(static)				\
  private(i2, i3)				\
  shared(i2min, i2max, i3min, i3max, vz, sim)
#endif  
      for(i3=i3min; i3<=i3max; i3++){
	for(i2=i2min; i2<=i2max; i2++){
	  vz[i3][i2][sim->nb-1] = vz[i3][i2][sim->nb];
	  vz[i3][i2][sim->nb-2] = vz[i3][i2][sim->nb+1];
	}
      }

    }else if(sim->order==8){
#ifdef _OPENMP
#pragma omp parallel for default(none)		\
  schedule(static)				\
  private(i2, i3)				\
  shared(i2min, i2max, i3min, i3max, vz, sim)
#endif  
      for(i3=i3min; i3<=i3max; i3++){
	for(i2=i2min; i2<=i2max; i2++){
	  vz[i3][i2][sim->nb-1] = vz[i3][i2][sim->nb];
	  vz[i3][i2][sim->nb-2] = vz[i3][i2][sim->nb+1];
	  vz[i3][i2][sim->nb-3] = vz[i3][i2][sim->nb+2];
	  vz[i3][i2][sim->nb-4] = vz[i3][i2][sim->nb+3];
	}
      }

    }//end if order
  }//end if freesurf
  
}


void fdtd_update_p(sim_t *sim, float ***p, float ***vz, float ***vx, float ***vy,
                   float ***memD1vz, float ***memD2vx, float ***memD3vy,
                   float ***xi1, float ***xi2, float ***xi3,
                   float ***kappa, float ***qinvmod)
{
  int i1, i2, i3, j1, j2, j3, k1, k2, k3;
  float D1vz, D2vx, D3vy, divv, relax_term, xi_old, xi_new;
  int i1min, i1max, i2min, i2max, i3min, i3max;
  float dt = sim->dt;
  float _d1 = 1./sim->d1;
  float _d2 = 1./sim->d2;
  float _d3 = 1./sim->d3;

  if(sim->order==4){
    i1min = 2;
    i1max = sim->n1pad-3;
    i2min = 2;
    i2max = sim->n2pad-3;
    i3min = (sim->n3>1)?2:0;
    i3max = (sim->n3>1)?(sim->n3pad-3):0;
  }else if(sim->order==8){
    i1min = 4;
    i1max = sim->n1pad-5;
    i2min = 4;
    i2max = sim->n2pad-5;
    i3min = (sim->n3>1)?4:0;
    i3max = (sim->n3>1)?(sim->n3pad-5):0;
  }
  
  if(sim->freesurf) i1min = sim->nb;
  
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3, D1vz, D2vx, D3vy, divv, 	\
  	 relax_term, xi_old, xi_new)					\
  shared(i1min, i1max, i2min, i2max, i3min, i3max, _d1, _d2, _d3, dt,	\
	 p, vz, vx, vy, memD1vz, memD2vx, memD3vy, xi1, xi2, xi3, 	\
	 kappa, qinvmod, sim)
#endif  
  for(i3=i3min; i3<=i3max; i3++) {
    for(i2=i2min; i2<=i2max; i2++) {
      for(i1=i1min; i1<=i1max; i1++) {
	if(sim->order==4){
	  D1vz = 1.125*(vz[i3][i2][i1]-vz[i3][i2][i1-1])
	    -0.041666666666666664*(vz[i3][i2][i1+1]-vz[i3][i2][i1-2]);
	  D2vx = 1.125*(vx[i3][i2][i1]-vx[i3][i2-1][i1])
	    -0.041666666666666664*(vx[i3][i2+1][i1]-vx[i3][i2-2][i1]);
	}else if(sim->order==8){
	  D1vz = 1.196289062500000*(vz[i3][i2][i1]-vz[i3][i2][i1-1])
	    -0.079752604166667*(vz[i3][i2][i1+1]-vz[i3][i2][i1-2])
	    +0.009570312500000*(vz[i3][i2][i1+2]-vz[i3][i2][i1-3])
	    -0.000697544642857*(vz[i3][i2][i1+3]-vz[i3][i2][i1-4]);
	  D2vx = 1.196289062500000*(vx[i3][i2][i1]-vx[i3][i2-1][i1])
	    -0.079752604166667*(vx[i3][i2+1][i1]-vx[i3][i2-2][i1])
	    +0.009570312500000*(vx[i3][i2+2][i1]-vx[i3][i2-3][i1])
	    -0.000697544642857*(vx[i3][i2+3][i1]-vx[i3][i2-4][i1]);
	}
	D1vz *= _d1;
	D2vx *= _d2;
      
	if(i1<sim->nb) {
	  memD1vz[i3][i2][i1] = sim->pmlb[i1]*memD1vz[i3][i2][i1] + sim->pmla[i1]*D1vz;
	  D1vz += memD1vz[i3][i2][i1];
	}else if(i1>=sim->n1pad-sim->nb){
	  j1 = sim->n1pad -1 - i1;
	  k1 = j1 + sim->nb;
	  memD1vz[i3][i2][k1] = sim->pmlb[j1]*memD1vz[i3][i2][k1] + sim->pmla[j1]*D1vz;
	  D1vz += memD1vz[i3][i2][k1];
	}
	if(i2<sim->nb) {
	  memD2vx[i3][i2][i1] = sim->pmlb[i2]*memD2vx[i3][i2][i1] + sim->pmla[i2]*D2vx;
	  D2vx += memD2vx[i3][i2][i1];
	}else if(i2>=sim->n2pad-sim->nb){
	  j2 = sim->n2pad-1-i2;
	  k2 = j2 + sim->nb;
	  memD2vx[i3][k2][i1] = sim->pmlb[j2]*memD2vx[i3][k2][i1] + sim->pmla[j2]*D2vx;
	  D2vx += memD2vx[i3][k2][i1];
	}

	if(sim->n3>1){
	  if(sim->order==4){
	    D3vy = 1.125*(vy[i3][i2][i1]-vy[i3-1][i2][i1])
	      -0.041666666666666664*(vy[i3+1][i2][i1]-vy[i3-2][i2][i1]);	  
	  }else if(sim->order==8){
	    D3vy = 1.196289062500000*(vy[i3][i2][i1]-vy[i3-1][i2][i1])
	      -0.079752604166667*(vy[i3+1][i2][i1]-vy[i3-2][i2][i1])
	      +0.009570312500000*(vy[i3+2][i2][i1]-vy[i3-3][i2][i1])
	      -0.000697544642857*(vy[i3+3][i2][i1]-vy[i3-4][i2][i1]);
	  }
	  D3vy *= _d3;
      
	  if(i3<sim->nb) {
	    memD3vy[i3][i2][i1] = sim->pmlb[i3]*memD3vy[i3][i2][i1] + sim->pmla[i3]*D3vy;
	    D3vy += memD3vy[i3][i2][i1];
	  }else if(i3>=sim->n3pad-sim->nb){
	    j3 = sim->n3pad-1-i3;
	    k3 = j3 + sim->nb;
	    memD3vy[k3][i2][i1] = sim->pmlb[j3]*memD3vy[k3][i2][i1] + sim->pmla[j3]*D3vy;
	    D3vy += memD3vy[k3][i2][i1];
	  }
	}else{
	  D3vy = 0.;
	}

        divv = D1vz + D2vx + D3vy;
        relax_term = 0.0f;
        if(sim->nrelax==3){
          xi_old = xi1[i3][i2][i1];
          xi_new = sim->sls_exp[0]*xi_old + sim->sls_1_exp[0]*divv;
          xi1[i3][i2][i1] = xi_new;
          relax_term += sim->sls_yl[0]*0.5f*(xi_old + xi_new);

          xi_old = xi2[i3][i2][i1];
          xi_new = sim->sls_exp[1]*xi_old + sim->sls_1_exp[1]*divv;
          xi2[i3][i2][i1] = xi_new;
          relax_term += sim->sls_yl[1]*0.5f*(xi_old + xi_new);

          xi_old = xi3[i3][i2][i1];
          xi_new = sim->sls_exp[2]*xi_old + sim->sls_1_exp[2]*divv;
          xi3[i3][i2][i1] = xi_new;
          relax_term += sim->sls_yl[2]*0.5f*(xi_old + xi_new);

          relax_term *= qinvmod[i3][i2][i1];
        }
        p[i3][i2][i1] -= dt*kappa[i3][i2][i1]*(divv - relax_term);
      }
    }
  }

  if(sim->freesurf){
    if(sim->order==4){
#ifdef _OPENMP
#pragma omp parallel for default(none)		\
  schedule(static)				\
  private(i2, i3)				\
  shared(i2min, i2max, i3min, i3max, p, sim)
#endif  
      for(i3=i3min; i3<=i3max; i3++){
	for(i2=i2min; i2<=i2max; i2++){
	  p[i3][i2][sim->nb] = 0;
	  p[i3][i2][sim->nb-1] = -p[i3][i2][sim->nb+1];
	}
      }
    }else if(sim->order==8){
#ifdef _OPENMP
#pragma omp parallel for default(none)		\
  schedule(static)				\
  private(i2, i3)				\
  shared(i2min, i2max, i3min, i3max, p, sim)
#endif  
      for(i3=i3min; i3<=i3max; i3++){
	for(i2=i2min; i2<=i2max; i2++){
	  p[i3][i2][sim->nb] = 0;
	  p[i3][i2][sim->nb-1] = -p[i3][i2][sim->nb+1];
	  p[i3][i2][sim->nb-2] = -p[i3][i2][sim->nb+2];
	  p[i3][i2][sim->nb-3] = -p[i3][i2][sim->nb+3];
	}
      }

    }//end if order
  }//end if freesurf

}
