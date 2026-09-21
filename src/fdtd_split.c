/* Finite-difference time-domain (FDTD) modelling kernel
 * PML-interior loop-split version.
 *
 * The innermost loops of the original fdtd.c carry PML-boundary branches
 * (if(i1<sim->nb), if(i2>=...), etc.) and the sim->order test on every
 * grid point.  For large models > 90 % of the padded grid is pure interior
 * where these branches always go the same way, wasting branch-prediction
 * resources and blocking auto-vectorisation.
 *
 * This file splits each FDTD step into one tight interior loop (no PML
 * branches, coefficients pre-loaded from arrays) followed by six face
 * loops covering the PML layers where the branches are perfectly
 * predictable.
 *
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

/* ------------------------------------------------------------------ */
/*  Velocity update  (split)                                          */
/* ------------------------------------------------------------------ */
void fdtd_update_v_split(sim_t *sim,
			 float ***p, float ***vz, float ***vx, float ***vy,
			 float ***memD1p, float ***memD2p, float ***memD3p,
			 float ***buz, float ***bux, float ***buy)
{
  int i1, i2, i3, j1, j2, j3, k1, k2, k3;
  float D1p, D2p, D3p;
  int i1min, i1max, i2min, i2max, i3min, i3max;
  int i1L, i1R, i2L, i2R, i3L, i3R;
  int has_interior;
  float dt = sim->dt;
  float _d1 = 1.f/sim->d1;
  float _d2 = 1.f/sim->d2;
  float _d3 = 1.f/sim->d3;

  /* FD coefficients loaded once (no order-branch in the inner loops) */
  float c0, c1, c2, c3;                  /* D1 / D2 */
  float d0, d1, d2, d3;                  /* D3 */
  if(sim->order==4){
    c0 = 1.125f;          c1 = -0.041666666666666664f;
    c2 = 0.0f;            c3 = 0.0f;
    d0 = 1.125f;          d1 = -0.041666666666666664f;
    d2 = 0.0f;            d3 = 0.0f;
  } else {
    c0 = 1.196289062500000f;  c1 = -0.079752604166667f;
    c2 = 0.009570312500000f;  c3 = -0.000697544642857f;
    d0 = 1.196289062500000f;  d1 = -0.079752604166667f;
    d2 = 0.009570312500000f;  d3 = -0.000697544642857f;
  }

  /* ---- overall bounds (same as original) ---- */
  if(sim->order==4){
    i1min = 1;       i1max = sim->n1pad-3;
    i2min = 1;       i2max = sim->n2pad-3;
    i3min = (sim->n3>1) ? 1 : 0;
    i3max = (sim->n3>1) ? (sim->n3pad-3) : 0;
  } else {
    i1min = 3;       i1max = sim->n1pad-5;
    i2min = 3;       i2max = sim->n2pad-5;
    i3min = (sim->n3>1) ? 3 : 0;
    i3max = (sim->n3>1) ? (sim->n3pad-5) : 0;
  }
  if(sim->freesurf) i1min = sim->nb;

  /* ---- interior bounds (no PML in any dimension) ---- */
  /* Right PML for update_v triggers at i1 >= n1pad-nb-1.
   * Interior therefore stops at n1pad-nb-2.  Same logic for i2, i3. */
  i1L = (i1min > sim->nb) ? i1min : sim->nb;
  i1R = (i1max < sim->n1pad - sim->nb - 2) ? i1max : (sim->n1pad - sim->nb - 2);
  i2L = (i2min > sim->nb) ? i2min : sim->nb;
  i2R = (i2max < sim->n2pad - sim->nb - 2) ? i2max : (sim->n2pad - sim->nb - 2);
  if(sim->n3>1){
    i3L = (i3min > sim->nb) ? i3min : sim->nb;
    i3R = (i3max < sim->n3pad - sim->nb - 2) ? i3max : (sim->n3pad - sim->nb - 2);
  } else {
    i3L = i3min;
    i3R = i3max;
  }
  has_interior = (i1L <= i1R && i2L <= i2R && i3L <= i3R);

  /* ================================================================ */
  /*  PART 1 — INTERIOR  (pure FD stencil, no PML, no 2D/3D branch)    */
  /* ================================================================ */
  if(has_interior){
    if(sim->n3>1){
      /* ----- 3D interior ----- */
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, D1p, D2p, D3p)					\
  shared(i1L, i1R, i2L, i2R, i3L, i3R, _d1, _d2, _d3, dt,		\
	 p, vz, vx, vy, buz, bux, buy, c0, c1, c2, c3, d0, d1, d2, d3)
#endif
      for(i3=i3L; i3<=i3R; i3++){
	for(i2=i2L; i2<=i2R; i2++){
	  for(i1=i1L; i1<=i1R; i1++){
	    D1p = c0*(p[i3][i2][i1+1]-p[i3][i2][i1])
	        + c1*(p[i3][i2][i1+2]-p[i3][i2][i1-1])
	        + c2*(p[i3][i2][i1+3]-p[i3][i2][i1-2])
	        + c3*(p[i3][i2][i1+4]-p[i3][i2][i1-3]);
	    D2p = c0*(p[i3][i2+1][i1]-p[i3][i2][i1])
	        + c1*(p[i3][i2+2][i1]-p[i3][i2-1][i1])
	        + c2*(p[i3][i2+3][i1]-p[i3][i2-2][i1])
	        + c3*(p[i3][i2+4][i1]-p[i3][i2-3][i1]);
	    D1p *= _d1;
	    D2p *= _d2;
	    vz[i3][i2][i1] -= dt*buz[i3][i2][i1]*D1p;
	    vx[i3][i2][i1] -= dt*bux[i3][i2][i1]*D2p;

	    D3p = d0*(p[i3+1][i2][i1]-p[i3][i2][i1])
	        + d1*(p[i3+2][i2][i1]-p[i3-1][i2][i1])
	        + d2*(p[i3+3][i2][i1]-p[i3-2][i2][i1])
	        + d3*(p[i3+4][i2][i1]-p[i3-3][i2][i1]);
	    D3p *= _d3;
	    vy[i3][i2][i1] -= dt*buy[i3][i2][i1]*D3p;
	  }
	}
      }
    } else {
      /* ----- 2D interior ----- */
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, D1p, D2p)					\
  shared(i1L, i1R, i2L, i2R, i3L, i3R, _d1, _d2, dt,			\
	 p, vz, vx, buz, bux, c0, c1, c2, c3)
#endif
      for(i3=i3L; i3<=i3R; i3++){
	for(i2=i2L; i2<=i2R; i2++){
	  for(i1=i1L; i1<=i1R; i1++){
	    D1p = c0*(p[i3][i2][i1+1]-p[i3][i2][i1])
	        + c1*(p[i3][i2][i1+2]-p[i3][i2][i1-1])
	        + c2*(p[i3][i2][i1+3]-p[i3][i2][i1-2])
	        + c3*(p[i3][i2][i1+4]-p[i3][i2][i1-3]);
	    D2p = c0*(p[i3][i2+1][i1]-p[i3][i2][i1])
	        + c1*(p[i3][i2+2][i1]-p[i3][i2-1][i1])
	        + c2*(p[i3][i2+3][i1]-p[i3][i2-2][i1])
	        + c3*(p[i3][i2+4][i1]-p[i3][i2-3][i1]);
	    D1p *= _d1;
	    D2p *= _d2;
	    vz[i3][i2][i1] -= dt*buz[i3][i2][i1]*D1p;
	    vx[i3][i2][i1] -= dt*bux[i3][i2][i1]*D2p;
	  }
	}
      }
    }
  } /* end interior */

  /* ================================================================ */
  /*  PART 2 — BOUNDARY FACES  (full PML logic, well-predicted)        */
  /* ================================================================ */

  /* --- Left i1 face --- */
  if(i1min <= i1L-1){
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3, D1p, D2p, D3p)		\
  shared(i1min, i1L, i2min, i2max, i3min, i3max, _d1, _d2, _d3, dt,	\
	 p, vz, vx, vy, memD1p, memD2p, memD3p, buz, bux, buy, sim,	\
	 c0, c1, c2, c3, d0, d1, d2, d3)
#endif
    for(i3=i3min; i3<=i3max; i3++){
      for(i2=i2min; i2<=i2max; i2++){
	for(i1=i1min; i1<=i1L-1; i1++){
	  if(sim->order==4){
	    D1p = 1.125*(p[i3][i2][i1+1]-p[i3][i2][i1])
	      -0.041666666666666664*(p[i3][i2][i1+2]-p[i3][i2][i1-1]);
	    D2p = 1.125*(p[i3][i2+1][i1]-p[i3][i2][i1])
	      -0.041666666666666664*(p[i3][i2+2][i1]-p[i3][i2-1][i1]);
	  } else {
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

	  if(i1<sim->nb){
	    memD1p[i3][i2][i1] = sim->pmlb_ph[i1]*memD1p[i3][i2][i1]
	                       + sim->pmla_ph[i1]*D1p;
	    D1p += memD1p[i3][i2][i1];
	  } else if(i1>=sim->n1pad-sim->nb-1){
	    j1 = sim->n1pad-2-i1;
	    k1 = j1 + sim->nb;
	    memD1p[i3][i2][k1] = sim->pmlb_ph[j1]*memD1p[i3][i2][k1]
	                       + sim->pmla_ph[j1]*D1p;
	    D1p += memD1p[i3][i2][k1];
	  }
	  if(i2<sim->nb){
	    memD2p[i3][i2][i1] = sim->pmlb_ph[i2]*memD2p[i3][i2][i1]
	                       + sim->pmla_ph[i2]*D2p;
	    D2p += memD2p[i3][i2][i1];
	  } else if(i2>=sim->n2pad-sim->nb-1){
	    j2 = sim->n2pad-2-i2;
	    k2 = j2 + sim->nb;
	    memD2p[i3][k2][i1] = sim->pmlb_ph[j2]*memD2p[i3][k2][i1]
	                       + sim->pmla_ph[j2]*D2p;
	    D2p += memD2p[i3][k2][i1];
	  }

	  vz[i3][i2][i1] -= dt*buz[i3][i2][i1]*D1p;
	  vx[i3][i2][i1] -= dt*bux[i3][i2][i1]*D2p;

	  if(sim->n3>1){
	    if(sim->order==4){
	      D3p = 1.125*(p[i3+1][i2][i1]-p[i3][i2][i1])
		-0.041666666666666664*(p[i3+2][i2][i1]-p[i3-1][i2][i1]);
	    } else {
	      D3p = 1.196289062500000*(p[i3+1][i2][i1]-p[i3][i2][i1])
		-0.079752604166667*(p[i3+2][i2][i1]-p[i3-1][i2][i1])
		+0.009570312500000*(p[i3+3][i2][i1]-p[i3-2][i2][i1])
		-0.000697544642857*(p[i3+4][i2][i1]-p[i3-3][i2][i1]);
	    }
	    D3p *= _d3;
	    if(i3<sim->nb){
	      memD3p[i3][i2][i1] = sim->pmlb_ph[i3]*memD3p[i3][i2][i1]
	                         + sim->pmla_ph[i3]*D3p;
	      D3p += memD3p[i3][i2][i1];
	    } else if(i3>=sim->n3pad-sim->nb-1){
	      j3 = sim->n3pad-2-i3;
	      k3 = j3 + sim->nb;
	      memD3p[k3][i2][i1] = sim->pmlb_ph[j3]*memD3p[k3][i2][i1]
	                         + sim->pmla_ph[j3]*D3p;
	      D3p += memD3p[k3][i2][i1];
	    }
	    vy[i3][i2][i1] -= dt*buy[i3][i2][i1]*D3p;
	  }
	}
      }
    }
  }

  /* --- Right i1 face --- */
  if(i1R+1 <= i1max){
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3, D1p, D2p, D3p)		\
  shared(i1R, i1max, i2min, i2max, i3min, i3max, _d1, _d2, _d3, dt,	\
	 p, vz, vx, vy, memD1p, memD2p, memD3p, buz, bux, buy, sim,	\
	 c0, c1, c2, c3, d0, d1, d2, d3)
#endif
    for(i3=i3min; i3<=i3max; i3++){
      for(i2=i2min; i2<=i2max; i2++){
	for(i1=i1R+1; i1<=i1max; i1++){
	  if(sim->order==4){
	    D1p = 1.125*(p[i3][i2][i1+1]-p[i3][i2][i1])
	      -0.041666666666666664*(p[i3][i2][i1+2]-p[i3][i2][i1-1]);
	    D2p = 1.125*(p[i3][i2+1][i1]-p[i3][i2][i1])
	      -0.041666666666666664*(p[i3][i2+2][i1]-p[i3][i2-1][i1]);
	  } else {
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

	  if(i1<sim->nb){
	    memD1p[i3][i2][i1] = sim->pmlb_ph[i1]*memD1p[i3][i2][i1]
	                       + sim->pmla_ph[i1]*D1p;
	    D1p += memD1p[i3][i2][i1];
	  } else if(i1>=sim->n1pad-sim->nb-1){
	    j1 = sim->n1pad-2-i1;
	    k1 = j1 + sim->nb;
	    memD1p[i3][i2][k1] = sim->pmlb_ph[j1]*memD1p[i3][i2][k1]
	                       + sim->pmla_ph[j1]*D1p;
	    D1p += memD1p[i3][i2][k1];
	  }
	  if(i2<sim->nb){
	    memD2p[i3][i2][i1] = sim->pmlb_ph[i2]*memD2p[i3][i2][i1]
	                       + sim->pmla_ph[i2]*D2p;
	    D2p += memD2p[i3][i2][i1];
	  } else if(i2>=sim->n2pad-sim->nb-1){
	    j2 = sim->n2pad-2-i2;
	    k2 = j2 + sim->nb;
	    memD2p[i3][k2][i1] = sim->pmlb_ph[j2]*memD2p[i3][k2][i1]
	                       + sim->pmla_ph[j2]*D2p;
	    D2p += memD2p[i3][k2][i1];
	  }

	  vz[i3][i2][i1] -= dt*buz[i3][i2][i1]*D1p;
	  vx[i3][i2][i1] -= dt*bux[i3][i2][i1]*D2p;

	  if(sim->n3>1){
	    if(sim->order==4){
	      D3p = 1.125*(p[i3+1][i2][i1]-p[i3][i2][i1])
		-0.041666666666666664*(p[i3+2][i2][i1]-p[i3-1][i2][i1]);
	    } else {
	      D3p = 1.196289062500000*(p[i3+1][i2][i1]-p[i3][i2][i1])
		-0.079752604166667*(p[i3+2][i2][i1]-p[i3-1][i2][i1])
		+0.009570312500000*(p[i3+3][i2][i1]-p[i3-2][i2][i1])
		-0.000697544642857*(p[i3+4][i2][i1]-p[i3-3][i2][i1]);
	    }
	    D3p *= _d3;
	    if(i3<sim->nb){
	      memD3p[i3][i2][i1] = sim->pmlb_ph[i3]*memD3p[i3][i2][i1]
	                         + sim->pmla_ph[i3]*D3p;
	      D3p += memD3p[i3][i2][i1];
	    } else if(i3>=sim->n3pad-sim->nb-1){
	      j3 = sim->n3pad-2-i3;
	      k3 = j3 + sim->nb;
	      memD3p[k3][i2][i1] = sim->pmlb_ph[j3]*memD3p[k3][i2][i1]
	                         + sim->pmla_ph[j3]*D3p;
	      D3p += memD3p[k3][i2][i1];
	    }
	    vy[i3][i2][i1] -= dt*buy[i3][i2][i1]*D3p;
	  }
	}
      }
    }
  }

  /* --- Left i2 face  (interior i1 range) --- */
  if(i2min <= i2L-1 && has_interior){
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3, D1p, D2p, D3p)		\
  shared(i1L, i1R, i2min, i2L, i3min, i3max, _d1, _d2, _d3, dt,	\
	 p, vz, vx, vy, memD1p, memD2p, memD3p, buz, bux, buy, sim,	\
	 c0, c1, c2, c3, d0, d1, d2, d3)
#endif
    for(i3=i3min; i3<=i3max; i3++){
      for(i2=i2min; i2<=i2L-1; i2++){
	for(i1=i1L; i1<=i1R; i1++){
	  if(sim->order==4){
	    D1p = 1.125*(p[i3][i2][i1+1]-p[i3][i2][i1])
	      -0.041666666666666664*(p[i3][i2][i1+2]-p[i3][i2][i1-1]);
	    D2p = 1.125*(p[i3][i2+1][i1]-p[i3][i2][i1])
	      -0.041666666666666664*(p[i3][i2+2][i1]-p[i3][i2-1][i1]);
	  } else {
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

	  /* i1 is interior — no D1p PML needed */
	  if(i2<sim->nb){
	    memD2p[i3][i2][i1] = sim->pmlb_ph[i2]*memD2p[i3][i2][i1]
	                       + sim->pmla_ph[i2]*D2p;
	    D2p += memD2p[i3][i2][i1];
	  } else if(i2>=sim->n2pad-sim->nb-1){
	    j2 = sim->n2pad-2-i2;
	    k2 = j2 + sim->nb;
	    memD2p[i3][k2][i1] = sim->pmlb_ph[j2]*memD2p[i3][k2][i1]
	                       + sim->pmla_ph[j2]*D2p;
	    D2p += memD2p[i3][k2][i1];
	  }

	  vz[i3][i2][i1] -= dt*buz[i3][i2][i1]*D1p;
	  vx[i3][i2][i1] -= dt*bux[i3][i2][i1]*D2p;

	  if(sim->n3>1){
	    if(sim->order==4){
	      D3p = 1.125*(p[i3+1][i2][i1]-p[i3][i2][i1])
		-0.041666666666666664*(p[i3+2][i2][i1]-p[i3-1][i2][i1]);
	    } else {
	      D3p = 1.196289062500000*(p[i3+1][i2][i1]-p[i3][i2][i1])
		-0.079752604166667*(p[i3+2][i2][i1]-p[i3-1][i2][i1])
		+0.009570312500000*(p[i3+3][i2][i1]-p[i3-2][i2][i1])
		-0.000697544642857*(p[i3+4][i2][i1]-p[i3-3][i2][i1]);
	    }
	    D3p *= _d3;
	    if(i3<sim->nb){
	      memD3p[i3][i2][i1] = sim->pmlb_ph[i3]*memD3p[i3][i2][i1]
	                         + sim->pmla_ph[i3]*D3p;
	      D3p += memD3p[i3][i2][i1];
	    } else if(i3>=sim->n3pad-sim->nb-1){
	      j3 = sim->n3pad-2-i3;
	      k3 = j3 + sim->nb;
	      memD3p[k3][i2][i1] = sim->pmlb_ph[j3]*memD3p[k3][i2][i1]
	                         + sim->pmla_ph[j3]*D3p;
	      D3p += memD3p[k3][i2][i1];
	    }
	    vy[i3][i2][i1] -= dt*buy[i3][i2][i1]*D3p;
	  }
	}
      }
    }
  }

  /* --- Right i2 face  (interior i1 range) --- */
  if(i2R+1 <= i2max && has_interior){
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3, D1p, D2p, D3p)		\
  shared(i1L, i1R, i2R, i2max, i3min, i3max, _d1, _d2, _d3, dt,	\
	 p, vz, vx, vy, memD1p, memD2p, memD3p, buz, bux, buy, sim,	\
	 c0, c1, c2, c3, d0, d1, d2, d3)
#endif
    for(i3=i3min; i3<=i3max; i3++){
      for(i2=i2R+1; i2<=i2max; i2++){
	for(i1=i1L; i1<=i1R; i1++){
	  if(sim->order==4){
	    D1p = 1.125*(p[i3][i2][i1+1]-p[i3][i2][i1])
	      -0.041666666666666664*(p[i3][i2][i1+2]-p[i3][i2][i1-1]);
	    D2p = 1.125*(p[i3][i2+1][i1]-p[i3][i2][i1])
	      -0.041666666666666664*(p[i3][i2+2][i1]-p[i3][i2-1][i1]);
	  } else {
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

	  /* i1 is interior — no D1p PML needed */
	  if(i2<sim->nb){
	    memD2p[i3][i2][i1] = sim->pmlb_ph[i2]*memD2p[i3][i2][i1]
	                       + sim->pmla_ph[i2]*D2p;
	    D2p += memD2p[i3][i2][i1];
	  } else if(i2>=sim->n2pad-sim->nb-1){
	    j2 = sim->n2pad-2-i2;
	    k2 = j2 + sim->nb;
	    memD2p[i3][k2][i1] = sim->pmlb_ph[j2]*memD2p[i3][k2][i1]
	                       + sim->pmla_ph[j2]*D2p;
	    D2p += memD2p[i3][k2][i1];
	  }

	  vz[i3][i2][i1] -= dt*buz[i3][i2][i1]*D1p;
	  vx[i3][i2][i1] -= dt*bux[i3][i2][i1]*D2p;

	  if(sim->n3>1){
	    if(sim->order==4){
	      D3p = 1.125*(p[i3+1][i2][i1]-p[i3][i2][i1])
		-0.041666666666666664*(p[i3+2][i2][i1]-p[i3-1][i2][i1]);
	    } else {
	      D3p = 1.196289062500000*(p[i3+1][i2][i1]-p[i3][i2][i1])
		-0.079752604166667*(p[i3+2][i2][i1]-p[i3-1][i2][i1])
		+0.009570312500000*(p[i3+3][i2][i1]-p[i3-2][i2][i1])
		-0.000697544642857*(p[i3+4][i2][i1]-p[i3-3][i2][i1]);
	    }
	    D3p *= _d3;
	    if(i3<sim->nb){
	      memD3p[i3][i2][i1] = sim->pmlb_ph[i3]*memD3p[i3][i2][i1]
	                         + sim->pmla_ph[i3]*D3p;
	      D3p += memD3p[i3][i2][i1];
	    } else if(i3>=sim->n3pad-sim->nb-1){
	      j3 = sim->n3pad-2-i3;
	      k3 = j3 + sim->nb;
	      memD3p[k3][i2][i1] = sim->pmlb_ph[j3]*memD3p[k3][i2][i1]
	                         + sim->pmla_ph[j3]*D3p;
	      D3p += memD3p[k3][i2][i1];
	    }
	    vy[i3][i2][i1] -= dt*buy[i3][i2][i1]*D3p;
	  }
	}
      }
    }
  }

  /* --- Left i3 face  (interior i1,i2 range, 3D only) --- */
  if(sim->n3>1 && i3min <= i3L-1 && has_interior){
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3, D1p, D2p, D3p)		\
  shared(i1L, i1R, i2L, i2R, i3min, i3L, _d1, _d2, _d3, dt,		\
	 p, vz, vx, vy, memD1p, memD2p, memD3p, buz, bux, buy, sim,	\
	 c0, c1, c2, c3, d0, d1, d2, d3)
#endif
    for(i3=i3min; i3<=i3L-1; i3++){
      for(i2=i2L; i2<=i2R; i2++){
	for(i1=i1L; i1<=i1R; i1++){
	  if(sim->order==4){
	    D1p = 1.125*(p[i3][i2][i1+1]-p[i3][i2][i1])
	      -0.041666666666666664*(p[i3][i2][i1+2]-p[i3][i2][i1-1]);
	    D2p = 1.125*(p[i3][i2+1][i1]-p[i3][i2][i1])
	      -0.041666666666666664*(p[i3][i2+2][i1]-p[i3][i2-1][i1]);
	  } else {
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

	  /* i1,i2 are interior — no D1p/D2p PML needed */

	  if(sim->order==4){
	    D3p = 1.125*(p[i3+1][i2][i1]-p[i3][i2][i1])
	      -0.041666666666666664*(p[i3+2][i2][i1]-p[i3-1][i2][i1]);
	  } else {
	    D3p = 1.196289062500000*(p[i3+1][i2][i1]-p[i3][i2][i1])
	      -0.079752604166667*(p[i3+2][i2][i1]-p[i3-1][i2][i1])
	      +0.009570312500000*(p[i3+3][i2][i1]-p[i3-2][i2][i1])
	      -0.000697544642857*(p[i3+4][i2][i1]-p[i3-3][i2][i1]);
	  }
	  D3p *= _d3;
	  if(i3<sim->nb){
	    memD3p[i3][i2][i1] = sim->pmlb_ph[i3]*memD3p[i3][i2][i1]
	                       + sim->pmla_ph[i3]*D3p;
	    D3p += memD3p[i3][i2][i1];
	  } else if(i3>=sim->n3pad-sim->nb-1){
	    j3 = sim->n3pad-2-i3;
	    k3 = j3 + sim->nb;
	    memD3p[k3][i2][i1] = sim->pmlb_ph[j3]*memD3p[k3][i2][i1]
	                       + sim->pmla_ph[j3]*D3p;
	    D3p += memD3p[k3][i2][i1];
	  }

	  vz[i3][i2][i1] -= dt*buz[i3][i2][i1]*D1p;
	  vx[i3][i2][i1] -= dt*bux[i3][i2][i1]*D2p;
	  vy[i3][i2][i1] -= dt*buy[i3][i2][i1]*D3p;
	}
      }
    }
  }

  /* --- Right i3 face  (interior i1,i2 range, 3D only) --- */
  if(sim->n3>1 && i3R+1 <= i3max && has_interior){
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3, D1p, D2p, D3p)		\
  shared(i1L, i1R, i2L, i2R, i3R, i3max, _d1, _d2, _d3, dt,		\
	 p, vz, vx, vy, memD1p, memD2p, memD3p, buz, bux, buy, sim,	\
	 c0, c1, c2, c3, d0, d1, d2, d3)
#endif
    for(i3=i3R+1; i3<=i3max; i3++){
      for(i2=i2L; i2<=i2R; i2++){
	for(i1=i1L; i1<=i1R; i1++){
	  if(sim->order==4){
	    D1p = 1.125*(p[i3][i2][i1+1]-p[i3][i2][i1])
	      -0.041666666666666664*(p[i3][i2][i1+2]-p[i3][i2][i1-1]);
	    D2p = 1.125*(p[i3][i2+1][i1]-p[i3][i2][i1])
	      -0.041666666666666664*(p[i3][i2+2][i1]-p[i3][i2-1][i1]);
	  } else {
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

	  /* i1,i2 are interior — no D1p/D2p PML needed */

	  if(sim->order==4){
	    D3p = 1.125*(p[i3+1][i2][i1]-p[i3][i2][i1])
	      -0.041666666666666664*(p[i3+2][i2][i1]-p[i3-1][i2][i1]);
	  } else {
	    D3p = 1.196289062500000*(p[i3+1][i2][i1]-p[i3][i2][i1])
	      -0.079752604166667*(p[i3+2][i2][i1]-p[i3-1][i2][i1])
	      +0.009570312500000*(p[i3+3][i2][i1]-p[i3-2][i2][i1])
	      -0.000697544642857*(p[i3+4][i2][i1]-p[i3-3][i2][i1]);
	  }
	  D3p *= _d3;
	  if(i3<sim->nb){
	    memD3p[i3][i2][i1] = sim->pmlb_ph[i3]*memD3p[i3][i2][i1]
	                       + sim->pmla_ph[i3]*D3p;
	    D3p += memD3p[i3][i2][i1];
	  } else if(i3>=sim->n3pad-sim->nb-1){
	    j3 = sim->n3pad-2-i3;
	    k3 = j3 + sim->nb;
	    memD3p[k3][i2][i1] = sim->pmlb_ph[j3]*memD3p[k3][i2][i1]
	                       + sim->pmla_ph[j3]*D3p;
	    D3p += memD3p[k3][i2][i1];
	  }

	  vz[i3][i2][i1] -= dt*buz[i3][i2][i1]*D1p;
	  vx[i3][i2][i1] -= dt*bux[i3][i2][i1]*D2p;
	  vy[i3][i2][i1] -= dt*buy[i3][i2][i1]*D3p;
	}
      }
    }
  }

  /* ================================================================ */
  /*  Free-surface clamping  (same as original)                        */
  /* ================================================================ */
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
    } else {
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
    }
  }
}


/* ------------------------------------------------------------------ */
/*  Pressure update  (split)                                           */
/* ------------------------------------------------------------------ */
void fdtd_update_p_split(sim_t *sim,
			 float ***p, float ***vz, float ***vx, float ***vy,
			 float ***memD1vz, float ***memD2vx, float ***memD3vy,
			 float ***xi1, float ***xi2, float ***xi3,
			 float ***kappa, float ***qinvmod)
{
  int i1, i2, i3, j1, j2, j3, k1, k2, k3;
  float D1vz, D2vx, D3vy, divv, relax_term, xi_old, xi_new;
  int i1min, i1max, i2min, i2max, i3min, i3max;
  int i1L, i1R, i2L, i2R, i3L, i3R;
  int has_interior;
  float dt = sim->dt;
  float _d1 = 1.f/sim->d1;
  float _d2 = 1.f/sim->d2;
  float _d3 = 1.f/sim->d3;

  /* FD coefficients loaded once */
  float c0, c1, c2, c3;
  float d0, d1, d2, d3;
  if(sim->order==4){
    c0 = 1.125f;          c1 = -0.041666666666666664f;
    c2 = 0.0f;            c3 = 0.0f;
    d0 = 1.125f;          d1 = -0.041666666666666664f;
    d2 = 0.0f;            d3 = 0.0f;
  } else {
    c0 = 1.196289062500000f;  c1 = -0.079752604166667f;
    c2 = 0.009570312500000f;  c3 = -0.000697544642857f;
    d0 = 1.196289062500000f;  d1 = -0.079752604166667f;
    d2 = 0.009570312500000f;  d3 = -0.000697544642857f;
  }

  /* ---- overall bounds ---- */
  if(sim->order==4){
    i1min = 2;       i1max = sim->n1pad-3;
    i2min = 2;       i2max = sim->n2pad-3;
    i3min = (sim->n3>1) ? 2 : 0;
    i3max = (sim->n3>1) ? (sim->n3pad-3) : 0;
  } else {
    i1min = 4;       i1max = sim->n1pad-5;
    i2min = 4;       i2max = sim->n2pad-5;
    i3min = (sim->n3>1) ? 4 : 0;
    i3max = (sim->n3>1) ? (sim->n3pad-5) : 0;
  }
  if(sim->freesurf) i1min = sim->nb;

  /* ---- interior bounds ---- */
  /* Right PML for update_p triggers at i1 >= n1pad-nb.
   * Interior stops at n1pad-nb-1. */
  i1L = (i1min > sim->nb) ? i1min : sim->nb;
  i1R = (i1max < sim->n1pad - sim->nb - 1) ? i1max : (sim->n1pad - sim->nb - 1);
  i2L = (i2min > sim->nb) ? i2min : sim->nb;
  i2R = (i2max < sim->n2pad - sim->nb - 1) ? i2max : (sim->n2pad - sim->nb - 1);
  if(sim->n3>1){
    i3L = (i3min > sim->nb) ? i3min : sim->nb;
    i3R = (i3max < sim->n3pad - sim->nb - 1) ? i3max : (sim->n3pad - sim->nb - 1);
  } else {
    i3L = i3min;
    i3R = i3max;
  }
  has_interior = (i1L <= i1R && i2L <= i2R && i3L <= i3R);

  /* ================================================================ */
  /*  PART 1 — INTERIOR                                                */
  /* ================================================================ */
  if(has_interior){
    if(sim->n3>1){
      /* ----- 3D interior ----- */
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, D1vz, D2vx, D3vy, divv, relax_term, xi_old, xi_new) \
  shared(i1L, i1R, i2L, i2R, i3L, i3R, _d1, _d2, _d3, dt,		\
	 p, vz, vx, vy, xi1, xi2, xi3, kappa, qinvmod, sim,		\
	 c0, c1, c2, c3, d0, d1, d2, d3)
#endif
      for(i3=i3L; i3<=i3R; i3++){
	for(i2=i2L; i2<=i2R; i2++){
	  for(i1=i1L; i1<=i1R; i1++){
	    D1vz = c0*(vz[i3][i2][i1]-vz[i3][i2][i1-1])
	         + c1*(vz[i3][i2][i1+1]-vz[i3][i2][i1-2])
	         + c2*(vz[i3][i2][i1+2]-vz[i3][i2][i1-3])
	         + c3*(vz[i3][i2][i1+3]-vz[i3][i2][i1-4]);
	    D2vx = c0*(vx[i3][i2][i1]-vx[i3][i2-1][i1])
	         + c1*(vx[i3][i2+1][i1]-vx[i3][i2-2][i1])
	         + c2*(vx[i3][i2+2][i1]-vx[i3][i2-3][i1])
	         + c3*(vx[i3][i2+3][i1]-vx[i3][i2-4][i1]);
	    D1vz *= _d1;
	    D2vx *= _d2;

	    D3vy = d0*(vy[i3][i2][i1]-vy[i3-1][i2][i1])
	         + d1*(vy[i3+1][i2][i1]-vy[i3-2][i2][i1])
	         + d2*(vy[i3+2][i2][i1]-vy[i3-3][i2][i1])
	         + d3*(vy[i3+3][i2][i1]-vy[i3-4][i2][i1]);
	    D3vy *= _d3;

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
    } else {
      /* ----- 2D interior ----- */
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, D1vz, D2vx, D3vy, divv, relax_term, xi_old, xi_new) \
  shared(i1L, i1R, i2L, i2R, i3L, i3R, _d1, _d2, dt,			\
	 p, vz, vx, xi1, xi2, xi3, kappa, qinvmod, sim,		\
	 c0, c1, c2, c3)
#endif
      for(i3=i3L; i3<=i3R; i3++){
	for(i2=i2L; i2<=i2R; i2++){
	  for(i1=i1L; i1<=i1R; i1++){
	    D1vz = c0*(vz[i3][i2][i1]-vz[i3][i2][i1-1])
	         + c1*(vz[i3][i2][i1+1]-vz[i3][i2][i1-2])
	         + c2*(vz[i3][i2][i1+2]-vz[i3][i2][i1-3])
	         + c3*(vz[i3][i2][i1+3]-vz[i3][i2][i1-4]);
	    D2vx = c0*(vx[i3][i2][i1]-vx[i3][i2-1][i1])
	         + c1*(vx[i3][i2+1][i1]-vx[i3][i2-2][i1])
	         + c2*(vx[i3][i2+2][i1]-vx[i3][i2-3][i1])
	         + c3*(vx[i3][i2+3][i1]-vx[i3][i2-4][i1]);
	    D1vz *= _d1;
	    D2vx *= _d2;

	    divv = D1vz + D2vx;
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
    }
  } /* end interior */

  /* ================================================================ */
  /*  PART 2 — BOUNDARY FACES                                          */
  /* ================================================================ */

  /* --- Left i1 face --- */
  if(i1min <= i1L-1){
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3, D1vz, D2vx, D3vy,	\
	  divv, relax_term, xi_old, xi_new)				\
  shared(i1min, i1L, i2min, i2max, i3min, i3max, _d1, _d2, _d3, dt,	\
	 p, vz, vx, vy, memD1vz, memD2vx, memD3vy,			\
	 xi1, xi2, xi3, kappa, qinvmod, sim,				\
	 c0, c1, c2, c3, d0, d1, d2, d3)
#endif
    for(i3=i3min; i3<=i3max; i3++){
      for(i2=i2min; i2<=i2max; i2++){
	for(i1=i1min; i1<=i1L-1; i1++){
	  if(sim->order==4){
	    D1vz = 1.125*(vz[i3][i2][i1]-vz[i3][i2][i1-1])
	      -0.041666666666666664*(vz[i3][i2][i1+1]-vz[i3][i2][i1-2]);
	    D2vx = 1.125*(vx[i3][i2][i1]-vx[i3][i2-1][i1])
	      -0.041666666666666664*(vx[i3][i2+1][i1]-vx[i3][i2-2][i1]);
	  } else {
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

	  if(i1<sim->nb){
	    memD1vz[i3][i2][i1] = sim->pmlb[i1]*memD1vz[i3][i2][i1]
	                        + sim->pmla[i1]*D1vz;
	    D1vz += memD1vz[i3][i2][i1];
	  } else if(i1>=sim->n1pad-sim->nb){
	    j1 = sim->n1pad-1-i1;
	    k1 = j1 + sim->nb;
	    memD1vz[i3][i2][k1] = sim->pmlb[j1]*memD1vz[i3][i2][k1]
	                        + sim->pmla[j1]*D1vz;
	    D1vz += memD1vz[i3][i2][k1];
	  }
	  if(i2<sim->nb){
	    memD2vx[i3][i2][i1] = sim->pmlb[i2]*memD2vx[i3][i2][i1]
	                        + sim->pmla[i2]*D2vx;
	    D2vx += memD2vx[i3][i2][i1];
	  } else if(i2>=sim->n2pad-sim->nb){
	    j2 = sim->n2pad-1-i2;
	    k2 = j2 + sim->nb;
	    memD2vx[i3][k2][i1] = sim->pmlb[j2]*memD2vx[i3][k2][i1]
	                        + sim->pmla[j2]*D2vx;
	    D2vx += memD2vx[i3][k2][i1];
	  }

	  if(sim->n3>1){
	    if(sim->order==4){
	      D3vy = 1.125*(vy[i3][i2][i1]-vy[i3-1][i2][i1])
		-0.041666666666666664*(vy[i3+1][i2][i1]-vy[i3-2][i2][i1]);
	    } else {
	      D3vy = 1.196289062500000*(vy[i3][i2][i1]-vy[i3-1][i2][i1])
		-0.079752604166667*(vy[i3+1][i2][i1]-vy[i3-2][i2][i1])
		+0.009570312500000*(vy[i3+2][i2][i1]-vy[i3-3][i2][i1])
		-0.000697544642857*(vy[i3+3][i2][i1]-vy[i3-4][i2][i1]);
	    }
	    D3vy *= _d3;
	    if(i3<sim->nb){
	      memD3vy[i3][i2][i1] = sim->pmlb[i3]*memD3vy[i3][i2][i1]
	                          + sim->pmla[i3]*D3vy;
	      D3vy += memD3vy[i3][i2][i1];
	    } else if(i3>=sim->n3pad-sim->nb){
	      j3 = sim->n3pad-1-i3;
	      k3 = j3 + sim->nb;
	      memD3vy[k3][i2][i1] = sim->pmlb[j3]*memD3vy[k3][i2][i1]
	                          + sim->pmla[j3]*D3vy;
	      D3vy += memD3vy[k3][i2][i1];
	    }
	  } else {
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
  }

  /* --- Right i1 face --- */
  if(i1R+1 <= i1max){
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3, D1vz, D2vx, D3vy,	\
	  divv, relax_term, xi_old, xi_new)				\
  shared(i1R, i1max, i2min, i2max, i3min, i3max, _d1, _d2, _d3, dt,	\
	 p, vz, vx, vy, memD1vz, memD2vx, memD3vy,			\
	 xi1, xi2, xi3, kappa, qinvmod, sim,				\
	 c0, c1, c2, c3, d0, d1, d2, d3)
#endif
    for(i3=i3min; i3<=i3max; i3++){
      for(i2=i2min; i2<=i2max; i2++){
	for(i1=i1R+1; i1<=i1max; i1++){
	  if(sim->order==4){
	    D1vz = 1.125*(vz[i3][i2][i1]-vz[i3][i2][i1-1])
	      -0.041666666666666664*(vz[i3][i2][i1+1]-vz[i3][i2][i1-2]);
	    D2vx = 1.125*(vx[i3][i2][i1]-vx[i3][i2-1][i1])
	      -0.041666666666666664*(vx[i3][i2+1][i1]-vx[i3][i2-2][i1]);
	  } else {
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

	  if(i1<sim->nb){
	    memD1vz[i3][i2][i1] = sim->pmlb[i1]*memD1vz[i3][i2][i1]
	                        + sim->pmla[i1]*D1vz;
	    D1vz += memD1vz[i3][i2][i1];
	  } else if(i1>=sim->n1pad-sim->nb){
	    j1 = sim->n1pad-1-i1;
	    k1 = j1 + sim->nb;
	    memD1vz[i3][i2][k1] = sim->pmlb[j1]*memD1vz[i3][i2][k1]
	                        + sim->pmla[j1]*D1vz;
	    D1vz += memD1vz[i3][i2][k1];
	  }
	  if(i2<sim->nb){
	    memD2vx[i3][i2][i1] = sim->pmlb[i2]*memD2vx[i3][i2][i1]
	                        + sim->pmla[i2]*D2vx;
	    D2vx += memD2vx[i3][i2][i1];
	  } else if(i2>=sim->n2pad-sim->nb){
	    j2 = sim->n2pad-1-i2;
	    k2 = j2 + sim->nb;
	    memD2vx[i3][k2][i1] = sim->pmlb[j2]*memD2vx[i3][k2][i1]
	                        + sim->pmla[j2]*D2vx;
	    D2vx += memD2vx[i3][k2][i1];
	  }

	  if(sim->n3>1){
	    if(sim->order==4){
	      D3vy = 1.125*(vy[i3][i2][i1]-vy[i3-1][i2][i1])
		-0.041666666666666664*(vy[i3+1][i2][i1]-vy[i3-2][i2][i1]);
	    } else {
	      D3vy = 1.196289062500000*(vy[i3][i2][i1]-vy[i3-1][i2][i1])
		-0.079752604166667*(vy[i3+1][i2][i1]-vy[i3-2][i2][i1])
		+0.009570312500000*(vy[i3+2][i2][i1]-vy[i3-3][i2][i1])
		-0.000697544642857*(vy[i3+3][i2][i1]-vy[i3-4][i2][i1]);
	    }
	    D3vy *= _d3;
	    if(i3<sim->nb){
	      memD3vy[i3][i2][i1] = sim->pmlb[i3]*memD3vy[i3][i2][i1]
	                          + sim->pmla[i3]*D3vy;
	      D3vy += memD3vy[i3][i2][i1];
	    } else if(i3>=sim->n3pad-sim->nb){
	      j3 = sim->n3pad-1-i3;
	      k3 = j3 + sim->nb;
	      memD3vy[k3][i2][i1] = sim->pmlb[j3]*memD3vy[k3][i2][i1]
	                          + sim->pmla[j3]*D3vy;
	      D3vy += memD3vy[k3][i2][i1];
	    }
	  } else {
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
  }

  /* --- Left i2 face (interior i1 range) --- */
  if(i2min <= i2L-1 && has_interior){
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3, D1vz, D2vx, D3vy,	\
	  divv, relax_term, xi_old, xi_new)				\
  shared(i1L, i1R, i2min, i2L, i3min, i3max, _d1, _d2, _d3, dt,	\
	 p, vz, vx, vy, memD1vz, memD2vx, memD3vy,			\
	 xi1, xi2, xi3, kappa, qinvmod, sim,				\
	 c0, c1, c2, c3, d0, d1, d2, d3)
#endif
    for(i3=i3min; i3<=i3max; i3++){
      for(i2=i2min; i2<=i2L-1; i2++){
	for(i1=i1L; i1<=i1R; i1++){
	  if(sim->order==4){
	    D1vz = 1.125*(vz[i3][i2][i1]-vz[i3][i2][i1-1])
	      -0.041666666666666664*(vz[i3][i2][i1+1]-vz[i3][i2][i1-2]);
	    D2vx = 1.125*(vx[i3][i2][i1]-vx[i3][i2-1][i1])
	      -0.041666666666666664*(vx[i3][i2+1][i1]-vx[i3][i2-2][i1]);
	  } else {
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

	  /* i1 is interior — no D1vz PML needed */
	  if(i2<sim->nb){
	    memD2vx[i3][i2][i1] = sim->pmlb[i2]*memD2vx[i3][i2][i1]
	                        + sim->pmla[i2]*D2vx;
	    D2vx += memD2vx[i3][i2][i1];
	  } else if(i2>=sim->n2pad-sim->nb){
	    j2 = sim->n2pad-1-i2;
	    k2 = j2 + sim->nb;
	    memD2vx[i3][k2][i1] = sim->pmlb[j2]*memD2vx[i3][k2][i1]
	                        + sim->pmla[j2]*D2vx;
	    D2vx += memD2vx[i3][k2][i1];
	  }

	  if(sim->n3>1){
	    if(sim->order==4){
	      D3vy = 1.125*(vy[i3][i2][i1]-vy[i3-1][i2][i1])
		-0.041666666666666664*(vy[i3+1][i2][i1]-vy[i3-2][i2][i1]);
	    } else {
	      D3vy = 1.196289062500000*(vy[i3][i2][i1]-vy[i3-1][i2][i1])
		-0.079752604166667*(vy[i3+1][i2][i1]-vy[i3-2][i2][i1])
		+0.009570312500000*(vy[i3+2][i2][i1]-vy[i3-3][i2][i1])
		-0.000697544642857*(vy[i3+3][i2][i1]-vy[i3-4][i2][i1]);
	    }
	    D3vy *= _d3;
	    if(i3<sim->nb){
	      memD3vy[i3][i2][i1] = sim->pmlb[i3]*memD3vy[i3][i2][i1]
	                          + sim->pmla[i3]*D3vy;
	      D3vy += memD3vy[i3][i2][i1];
	    } else if(i3>=sim->n3pad-sim->nb){
	      j3 = sim->n3pad-1-i3;
	      k3 = j3 + sim->nb;
	      memD3vy[k3][i2][i1] = sim->pmlb[j3]*memD3vy[k3][i2][i1]
	                          + sim->pmla[j3]*D3vy;
	      D3vy += memD3vy[k3][i2][i1];
	    }
	  } else {
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
  }

  /* --- Right i2 face (interior i1 range) --- */
  if(i2R+1 <= i2max && has_interior){
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3, D1vz, D2vx, D3vy,	\
	  divv, relax_term, xi_old, xi_new)				\
  shared(i1L, i1R, i2R, i2max, i3min, i3max, _d1, _d2, _d3, dt,	\
	 p, vz, vx, vy, memD1vz, memD2vx, memD3vy,			\
	 xi1, xi2, xi3, kappa, qinvmod, sim,				\
	 c0, c1, c2, c3, d0, d1, d2, d3)
#endif
    for(i3=i3min; i3<=i3max; i3++){
      for(i2=i2R+1; i2<=i2max; i2++){
	for(i1=i1L; i1<=i1R; i1++){
	  if(sim->order==4){
	    D1vz = 1.125*(vz[i3][i2][i1]-vz[i3][i2][i1-1])
	      -0.041666666666666664*(vz[i3][i2][i1+1]-vz[i3][i2][i1-2]);
	    D2vx = 1.125*(vx[i3][i2][i1]-vx[i3][i2-1][i1])
	      -0.041666666666666664*(vx[i3][i2+1][i1]-vx[i3][i2-2][i1]);
	  } else {
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

	  /* i1 is interior — no D1vz PML needed */
	  if(i2<sim->nb){
	    memD2vx[i3][i2][i1] = sim->pmlb[i2]*memD2vx[i3][i2][i1]
	                        + sim->pmla[i2]*D2vx;
	    D2vx += memD2vx[i3][i2][i1];
	  } else if(i2>=sim->n2pad-sim->nb){
	    j2 = sim->n2pad-1-i2;
	    k2 = j2 + sim->nb;
	    memD2vx[i3][k2][i1] = sim->pmlb[j2]*memD2vx[i3][k2][i1]
	                        + sim->pmla[j2]*D2vx;
	    D2vx += memD2vx[i3][k2][i1];
	  }

	  if(sim->n3>1){
	    if(sim->order==4){
	      D3vy = 1.125*(vy[i3][i2][i1]-vy[i3-1][i2][i1])
		-0.041666666666666664*(vy[i3+1][i2][i1]-vy[i3-2][i2][i1]);
	    } else {
	      D3vy = 1.196289062500000*(vy[i3][i2][i1]-vy[i3-1][i2][i1])
		-0.079752604166667*(vy[i3+1][i2][i1]-vy[i3-2][i2][i1])
		+0.009570312500000*(vy[i3+2][i2][i1]-vy[i3-3][i2][i1])
		-0.000697544642857*(vy[i3+3][i2][i1]-vy[i3-4][i2][i1]);
	    }
	    D3vy *= _d3;
	    if(i3<sim->nb){
	      memD3vy[i3][i2][i1] = sim->pmlb[i3]*memD3vy[i3][i2][i1]
	                          + sim->pmla[i3]*D3vy;
	      D3vy += memD3vy[i3][i2][i1];
	    } else if(i3>=sim->n3pad-sim->nb){
	      j3 = sim->n3pad-1-i3;
	      k3 = j3 + sim->nb;
	      memD3vy[k3][i2][i1] = sim->pmlb[j3]*memD3vy[k3][i2][i1]
	                          + sim->pmla[j3]*D3vy;
	      D3vy += memD3vy[k3][i2][i1];
	    }
	  } else {
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
  }

  /* --- Left i3 face (interior i1,i2 range, 3D only) --- */
  if(sim->n3>1 && i3min <= i3L-1 && has_interior){
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3, D1vz, D2vx, D3vy,	\
	  divv, relax_term, xi_old, xi_new)				\
  shared(i1L, i1R, i2L, i2R, i3min, i3L, _d1, _d2, _d3, dt,		\
	 p, vz, vx, vy, memD1vz, memD2vx, memD3vy,			\
	 xi1, xi2, xi3, kappa, qinvmod, sim,				\
	 c0, c1, c2, c3, d0, d1, d2, d3)
#endif
    for(i3=i3min; i3<=i3L-1; i3++){
      for(i2=i2L; i2<=i2R; i2++){
	for(i1=i1L; i1<=i1R; i1++){
	  if(sim->order==4){
	    D1vz = 1.125*(vz[i3][i2][i1]-vz[i3][i2][i1-1])
	      -0.041666666666666664*(vz[i3][i2][i1+1]-vz[i3][i2][i1-2]);
	    D2vx = 1.125*(vx[i3][i2][i1]-vx[i3][i2-1][i1])
	      -0.041666666666666664*(vx[i3][i2+1][i1]-vx[i3][i2-2][i1]);
	  } else {
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

	  /* i1,i2 are interior — no D1vz/D2vx PML needed */

	  if(sim->order==4){
	    D3vy = 1.125*(vy[i3][i2][i1]-vy[i3-1][i2][i1])
	      -0.041666666666666664*(vy[i3+1][i2][i1]-vy[i3-2][i2][i1]);
	  } else {
	    D3vy = 1.196289062500000*(vy[i3][i2][i1]-vy[i3-1][i2][i1])
	      -0.079752604166667*(vy[i3+1][i2][i1]-vy[i3-2][i2][i1])
	      +0.009570312500000*(vy[i3+2][i2][i1]-vy[i3-3][i2][i1])
	      -0.000697544642857*(vy[i3+3][i2][i1]-vy[i3-4][i2][i1]);
	  }
	  D3vy *= _d3;
	  if(i3<sim->nb){
	    memD3vy[i3][i2][i1] = sim->pmlb[i3]*memD3vy[i3][i2][i1]
	                        + sim->pmla[i3]*D3vy;
	    D3vy += memD3vy[i3][i2][i1];
	  } else if(i3>=sim->n3pad-sim->nb){
	    j3 = sim->n3pad-1-i3;
	    k3 = j3 + sim->nb;
	    memD3vy[k3][i2][i1] = sim->pmlb[j3]*memD3vy[k3][i2][i1]
	                        + sim->pmla[j3]*D3vy;
	    D3vy += memD3vy[k3][i2][i1];
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
  }

  /* --- Right i3 face (interior i1,i2 range, 3D only) --- */
  if(sim->n3>1 && i3R+1 <= i3max && has_interior){
#ifdef _OPENMP
#pragma omp parallel for default(none)					\
  schedule(static)							\
  private(i1, i2, i3, j1, j2, j3, k1, k2, k3, D1vz, D2vx, D3vy,	\
	  divv, relax_term, xi_old, xi_new)				\
  shared(i1L, i1R, i2L, i2R, i3R, i3max, _d1, _d2, _d3, dt,		\
	 p, vz, vx, vy, memD1vz, memD2vx, memD3vy,			\
	 xi1, xi2, xi3, kappa, qinvmod, sim,				\
	 c0, c1, c2, c3, d0, d1, d2, d3)
#endif
    for(i3=i3R+1; i3<=i3max; i3++){
      for(i2=i2L; i2<=i2R; i2++){
	for(i1=i1L; i1<=i1R; i1++){
	  if(sim->order==4){
	    D1vz = 1.125*(vz[i3][i2][i1]-vz[i3][i2][i1-1])
	      -0.041666666666666664*(vz[i3][i2][i1+1]-vz[i3][i2][i1-2]);
	    D2vx = 1.125*(vx[i3][i2][i1]-vx[i3][i2-1][i1])
	      -0.041666666666666664*(vx[i3][i2+1][i1]-vx[i3][i2-2][i1]);
	  } else {
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

	  /* i1,i2 are interior — no D1vz/D2vx PML needed */

	  if(sim->order==4){
	    D3vy = 1.125*(vy[i3][i2][i1]-vy[i3-1][i2][i1])
	      -0.041666666666666664*(vy[i3+1][i2][i1]-vy[i3-2][i2][i1]);
	  } else {
	    D3vy = 1.196289062500000*(vy[i3][i2][i1]-vy[i3-1][i2][i1])
	      -0.079752604166667*(vy[i3+1][i2][i1]-vy[i3-2][i2][i1])
	      +0.009570312500000*(vy[i3+2][i2][i1]-vy[i3-3][i2][i1])
	      -0.000697544642857*(vy[i3+3][i2][i1]-vy[i3-4][i2][i1]);
	  }
	  D3vy *= _d3;
	  if(i3<sim->nb){
	    memD3vy[i3][i2][i1] = sim->pmlb[i3]*memD3vy[i3][i2][i1]
	                        + sim->pmla[i3]*D3vy;
	    D3vy += memD3vy[i3][i2][i1];
	  } else if(i3>=sim->n3pad-sim->nb){
	    j3 = sim->n3pad-1-i3;
	    k3 = j3 + sim->nb;
	    memD3vy[k3][i2][i1] = sim->pmlb[j3]*memD3vy[k3][i2][i1]
	                        + sim->pmla[j3]*D3vy;
	    D3vy += memD3vy[k3][i2][i1];
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
  }

  /* ================================================================ */
  /*  Free-surface clamping  (same as original)                        */
  /* ================================================================ */
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
    } else {
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
    }
  }
}