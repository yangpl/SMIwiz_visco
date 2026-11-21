/* 2D/3D seismic modelling, RTM and FWI code
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
#include "modelling.h"

void computing_box_init(acq_t *acq, sim_t *sim)
/*< specify computing box to do simulation efficiently >*/
{
  int isrc,it,nxe,nze,nye,shift;
  float t;

  if(!sim->ibox) return;
  /* Cover source interpolation, the FD stencil, and coordinate rounding. */
  shift = sim->ri + sim->order/2 + 1;

  sim->i1min=alloc1int(sim->nt);
  sim->i1max=alloc1int(sim->nt);
  sim->i2min=alloc1int(sim->nt);
  sim->i2max=alloc1int(sim->nt);
  sim->i3min=alloc1int(sim->nt);
  sim->i3max=alloc1int(sim->nt);
  sim->i1min[0]=acq->src_i1[0];
  sim->i1max[0]=acq->src_i1[0];
  sim->i2min[0]=acq->src_i2[0];
  sim->i2max[0]=acq->src_i2[0];
  if(sim->n3>1) {
    sim->i3min[0]=acq->src_i3[0];
    sim->i3max[0]=acq->src_i3[0];
  }else{
    sim->i3min[0]=0;
    sim->i3max[0]=0;	
  }
  for(isrc=0; isrc<acq->nsrc; isrc++){
    sim->i1min[0]=MIN(sim->i1min[0],acq->src_i1[isrc]);
    sim->i1max[0]=MAX(sim->i1max[0],acq->src_i1[isrc]);
    sim->i2min[0]=MIN(sim->i2min[0],acq->src_i2[isrc]);
    sim->i2max[0]=MAX(sim->i2max[0],acq->src_i2[isrc]);
    if(sim->n3>1){
      sim->i3min[0]=MIN(sim->i3min[0],acq->src_i3[isrc]);
      sim->i3max[0]=MAX(sim->i3max[0],acq->src_i3[isrc]);
    }
  }
  for(it=1; it<sim->nt; it++){
    t=it*sim->dt;
    nze=NINT(sim->vmax*t/sim->d1)+shift;
    nxe=NINT(sim->vmax*t/sim->d2)+shift;
    sim->i1min[it]=MAX(sim->i1min[0]-nze,0);
    sim->i1max[it]=MIN(sim->i1max[0]+nze,sim->n1pad-1);
    sim->i2min[it]=MAX(sim->i2min[0]-nxe,0);
    sim->i2max[it]=MIN(sim->i2max[0]+nxe,sim->n2pad-1);
      
    if(sim->n3>1){
      nye=NINT(sim->vmax*t/sim->d3)+shift;
      sim->i3min[it]=MAX(sim->i3min[0]-nye,0);
      sim->i3max[it]=MIN(sim->i3max[0]+nye,sim->n3pad-1);
    }else
      sim->i3min[it]=sim->i3max[it]=0;
  }

  /* Expand step zero last: later bounds above use the source extrema. */
  sim->i1min[0]=MAX(sim->i1min[0]-shift,0);
  sim->i1max[0]=MIN(sim->i1max[0]+shift,sim->n1pad-1);
  sim->i2min[0]=MAX(sim->i2min[0]-shift,0);
  sim->i2max[0]=MIN(sim->i2max[0]+shift,sim->n2pad-1);
  if(sim->n3>1){
    sim->i3min[0]=MAX(sim->i3min[0]-shift,0);
    sim->i3max[0]=MIN(sim->i3max[0]+shift,sim->n3pad-1);
  }
}

void computing_box_free(sim_t *sim)
{
  if(!sim->ibox) return;
  free1int(sim->i1min);
  free1int(sim->i1max);
  free1int(sim->i2min);
  free1int(sim->i2max);
  free1int(sim->i3min);
  free1int(sim->i3max);
  sim->i1min = sim->i1max = NULL;
  sim->i2min = sim->i2max = NULL;
  sim->i3min = sim->i3max = NULL;
}

