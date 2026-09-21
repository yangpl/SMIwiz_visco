#ifndef _sim_h_
#define _sim_h_

typedef struct {
  int mode;//0=modelling;1=FWI, 2=RTM, 3=linearized waveform inversion, 4=FWI gradient, 5=source inversion
  int order;//order of FD scheme
  int nt;//number of time steps
  int nsnap;//maximum number of Tapenade binomial checkpoints
  int n1, n2, n3, nb;//number grid points in 1st, 2nd and 3rd coorindate, number of ABC layers
  int n1pad, n2pad, n3pad;//model dimensions 
  int n123, n123pad;
  float d1, d2, d3;//grid spacing
  float volume;//cell volume
  float dt;//temporal sampling
  float freq;//dominant frequency for PML
  float *stf;//source time function
  int eachopt;//stf for each shot
  
  int freesurf;//1, stress-free surface condition;0, no free surface
  float cfl;//CFL number for stability
  float vmax, vmin;//maximum and minimum velocity
  float vphasemax;//maximum phase velocity considering VTI anisotropy
  float rhomax, rhomin;//maximum and minimum density

  int ri;//radius for Kaiser windowed sinc interpolation, 2*ri+1 points in total    
  float *pmla, *pmlb;//CPML damping factor
  float *pmla_ph, *pmlb_ph;//CPML damping factor with half grid shift
  
  float ***vp, ***rho, ***qinv;//model of original size
  float ***vpmod, ***rhomod, ***qinvmod;//extended vp, rho, qinv
  float ***kappa, ***buz, ***bux, ***buy;//extended bulk modulus and buoyancy=1/rho
  /* Optional standard-linear-solid attenuation.  nrelax=0 preserves the
   * original acoustic propagator; nrelax=3 activates the three-memory-
   * variable discretization in main.tex. */
  int nrelax;
  float sls_wl[3];
  float sls_yl[3];
  float sls_exp[3];
  float sls_1_exp[3];
  float ***xi1, ***xi2, ***xi3;//forward viscoacoustic memory variables
  float ***xi1b, ***xi2b, ***xi3b;//adjoint viscoacoustic memory variables

  //forward field and PML variables
  float ***vz, ***vx, ***vy, ***p;
  float ***memD1p, ***memD2p, ***memD3p;
  float ***memD1vz, ***memD2vx, ***memD3vy;
  
  //adjoint field and PML variables
  float ***vzb, ***vxb, ***vyb, ***pb;
  float ***memD1pb, ***memD2pb, ***memD3pb;
  float ***memD1vzb, ***memD2vxb, ***memD3vyb;
    
  //incident field and PML variables (used in RWI and LSRTM for scattering field)
  float ***vz0, ***vx0, ***vy0, ***p0;
  float ***memD1p0, ***memD2p0, ***memD3p0;
  float ***memD1vz0, ***memD2vx0, ***memD3vy0;

  float **dobs, **dcal, **dres;//observed, calculated and residual data
  
  //data weighting and muting option
  int muteopt;//mute options

  int itcheck;//at step itcheck, check wavefield snapshot

} sim_t;//simulator for forward and adjoint equations

#endif
