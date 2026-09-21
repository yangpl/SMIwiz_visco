#ifndef _fwi_h_
#define _fwi_h_

typedef struct {
  int n; //total number of unknowns in FWI
  int family;//1=vp-rho; 2=vp-ip
  int npar;//number of parameters to invert by FWI
  int *idxpar;//index of the inversion parameters
  float rhomin, rhomax, vpmin, vpmax;
  float qinvmin, qinvmax;

  float **bathy; //bathymetry to prescribe water bottom
  int **ibathy;//index of bathymetry at depth z

  int niter; //maximum number of iterations
  int niter_inner; //maximum number of iterations in inner loops
  int iter; //iteration index
  int restart;//restart iterations
  double fcost, fcost_dat;
  
  float alpha;//scaling factor for misfit function
  int firstgrad;//if first gradient computation, firstgrad=1, otherwise firstgrad=0
  int isrcpershot;// estimate source per shot or not
  int objopt;//0=L2, 1=AWI
} fwi_t;

#endif
