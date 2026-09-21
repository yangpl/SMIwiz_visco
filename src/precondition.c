/* Implicit reparameterization preconditioner for FWI.
 *
 * P = D*[eta*S*S^T + (1-eta)*I]*D is SPD for eta<1. S is the existing
 * separable triangle smoother and D is a positive depth-weighting diagonal.
 *----------------------------------------------------------------------------
 *  Copyright (c) Pengliang Yang, 2026, Laoshan Laboratory, China
 *  Copyright (c) Pengliang Yang, 2020, Harbin Institute of Technology, China
 *  Copyright (c) Pengliang Yang, 2018, University Grenoble Alpes, France
 *  Homepage: https://yangpl.wordpress.com
 *  E-mail: ypl.2100@gmail.com
 *--------------------------------------------------------------------------*/
#include "cstd.h"
#include "sim.h"
#include "fwi.h"

extern int iproc;
void triangle_smoothing(float ***mod, int n1, int n2, int n3,
                        int r1, int r2, int r3);

void precondition(sim_t *sim, fwi_t *fwi, float *x)
/*< apply the implicit-reparameterization preconditioner >*/
{
  int i1, i2, i3, ipar, j;
  int r1, r2, r3, preco_depth;
  float eta, depth_floor, depth_power, *raw;
  float ***smooth;

  if(!getparint("preco_r1", &r1)) r1 = MAX(1, NINT(0.05f*sim->n1));
  if(!getparint("preco_r2", &r2)) r2 = MAX(1, NINT(0.05f*sim->n2));
  if(!getparint("preco_r3", &r3)) r3 = MAX(1, NINT(0.05f*sim->n3));
  if(!getparint("preco_depth", &preco_depth)) preco_depth = 1;
  if(!getparfloat("preco_depth_floor", &depth_floor)) depth_floor = 0.1f;
  if(!getparfloat("preco_depth_power", &depth_power)) depth_power = 1.0f;
  if(r1<1 || r1>sim->n1 || r2<1 || r2>sim->n2 || r3<1 || r3>sim->n3)
    err("preco_r1, preco_r2, and preco_r3 must be within model dimensions");
  if(preco_depth!=0 && preco_depth!=1)
    err("preco_depth must be 0 or 1");
  if(!(depth_floor>0.0f && depth_floor<=1.0f && depth_power>0.0f))
    err("preco_depth_floor must be in (0,1] and preco_depth_power must be positive");

  eta = exp(-(float)fwi->iter/(float)fwi->niter);
  if(iproc==0 && fwi->iter==0)
    printf("implicit preconditioner: radii=[%d,%d,%d] depth=%d\n",
           r1, r2, r3, preco_depth);
  raw = alloc1float(sim->n123);
  smooth = alloc3float(sim->n1, sim->n2, sim->n3);
  for(ipar=0; ipar<fwi->npar; ipar++){
    memcpy(raw, &x[ipar*sim->n123], (size_t)sim->n123*sizeof(float));
    if(preco_depth){
      for(i3=0; i3<sim->n3; i3++)
        for(i2=0; i2<sim->n2; i2++)
	  for(i1=0; i1<sim->n1; i1++){
	    float z = sim->n1>1 ? (float)i1/(float)(sim->n1-1) : 1.0f;
	    float depth = depth_floor+(1.0f-depth_floor)*powf(z, depth_power);
	    raw[i1+sim->n1*(i2+sim->n2*i3)] *= depth;
	  }
    }
    memcpy(smooth[0][0], raw, (size_t)sim->n123*sizeof(float));
    /* triangle smoothing is self-adjoint on this uniform grid, i.e. 
     * S=box car smoothing,  we apply also depth preconditioning D, so 
     * P = D^{1/2}[eta* S*S^T + (1-eta)*I.]D^{1/2} */
    triangle_smoothing(smooth, sim->n1, sim->n2, sim->n3, r1, r2, r3);
    for(i3=0; i3<sim->n3; i3++){
      for(i2=0; i2<sim->n2; i2++){
	for(i1=0; i1<sim->n1; i1++){
	  float z = sim->n1>1 ? (float)i1/(float)(sim->n1-1) : 1.0f;
	  float depth = preco_depth
	    ? depth_floor+(1.0f-depth_floor)*powf(z, depth_power) : 1.0f;
	  j = i1 + sim->n1*(i2 + sim->n2*(i3 + sim->n3*ipar));
	  x[j] = depth*(eta*smooth[i3][i2][i1]
	       +(1.0f-eta)*raw[i1+sim->n1*(i2+sim->n2*i3)]);
	  if(i1<=fwi->ibathy[i3][i2]) x[j] = 0.0f;
	}
      }
    }
  }
  free3float(smooth);
  free1float(raw);
}
