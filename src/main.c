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
#include <mpi.h>

int iproc, nproc, ierr;

acq_t *acq;
sim_t *sim;

void acq_init(sim_t *sim, acq_t *acq);
void acq_free(sim_t *sim, acq_t *acq);

void read_data(sim_t *sim, acq_t *acq);
void setup_data_mask(acq_t *acq, sim_t *sim);

void visco_sls_fit(int L, float fmin, float fmax, float Q0, float *wl, float *yl);
void do_modelling(sim_t *sim, acq_t *acq);
void do_fwi(sim_t *sim, acq_t *acq);
void do_rtm(sim_t *sim, acq_t *acq);
void do_lsm(sim_t *sim, acq_t *acq);
void do_invert_source(sim_t *sim, acq_t *acq);

int main(int argc, char* argv[])
{
  char current_time[128];
  time_t      t;
  struct tm*  ptm;
  char *stffile, *vpfile, *rhofile, *qinvfile;
  FILE *fp;
    
  /* When stdout/stderr are redirected to log files, force line-buffered output so
   * long forward/adjoint solves still emit progress messages immediately. */
  setvbuf(stdout, NULL, _IOLBF, 0);
  setvbuf(stderr, NULL, _IOLBF, 0);

  MPI_Init(&argc, &argv);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  
  initargs(argc, argv);  
  acq = (acq_t *)malloc(sizeof(acq_t));
  sim = (sim_t *)malloc(sizeof(sim_t));
  if(acq==NULL || sim==NULL) err("cannot allocate main structures");
  
  if(!getparint("mode", &sim->mode)) sim->mode=0;
  if(sim->mode!=0 && sim->mode!=1 && sim->mode!=2 && sim->mode!=3
     && sim->mode!=4 && sim->mode!=5)
    err("mode must be 0 (modelling), 1 (FWI), 2 (RTM), 3 (linearized inversion), 4 (FWI gradient), or 5 (source inversion)");
  if(iproc==0){
    t = time(NULL);
    ptm = localtime(&t);
    strftime(current_time, 128, "%d-%b-%Y %H:%M:%S", ptm);
    printf("  Current date and time: %s\n", current_time);
    printf("=====================================================\n");
    printf("   Welcome to SMIwiz software: an integrated toolbox \n");
    printf("   for seismic modeling, RTM imaging, linearized and \n");
    printf("         nonlinear full waveform inversion           \n");
    printf("             Copyright (c) Pengliang Yang            \n");
    printf("             E-mail: ypl.2100@gmail.com              \n");
    printf("=====================================================\n");
    if(sim->mode==0) printf(" Forward modeling \n");
    else if(sim->mode==1) printf(" FWI in the time domain \n");
    else if(sim->mode==2) printf(" RTM in the time domain \n");
    else if(sim->mode==3) printf(" LSRTM (TLM/adjoint) \n");
    else if(sim->mode==4) printf(" FWI gradient \n");    
    else if(sim->mode==5) printf(" Source inversion \n");
    printf("=====================================================\n");
  }
  
  //=========== specify parameters for simulation ============
  if(!getparint("nt", &sim->nt)) err("must give nt= "); //total number of time steps
  if(!getparint("nsnap", &sim->nsnap)) sim->nsnap = 20;//number of snapshots for checkpointing
  if(!getparfloat("dt",&sim->dt)) err("must give dt= "); //temporal sampling
  if(!getparint("nb", &sim->nb)) sim->nb = 20;   //number of layers for PML absorbing boundary
  if(!getparint("n1",&sim->n1)) err("must give n1= for FD grid");
  if(!getparint("n2",&sim->n2)) err("must give n2= for FD grid");
  if(!getparfloat("d1",&sim->d1)) err("must give d1= for FD grid"); 
  if(!getparfloat("d2",&sim->d2)) err("must give d2= for FD grid");
  if(sim->nt<=0) err("nt must be positive");
  if(sim->nt>1 && (sim->nsnap<1 || sim->nsnap>98))
    err("nsnap must be in [1,98] when nt>1");
  if(sim->nt==1 && (sim->nsnap<0 || sim->nsnap>98))
    err("nsnap must be in [0,98] when nt=1");
  if(sim->dt<=0) err("dt must be positive");
  if(sim->n1<=0 || sim->n2<=0) err("n1 and n2 must be positive");
  if(sim->d1<=0 || sim->d2<=0) err("d1 and d2 must be positive");
  if(sim->nb<0) err("nb must be nonnegative");
  sim->n1pad = sim->n1+2*sim->nb;
  sim->n2pad = sim->n2+2*sim->nb;
  if(!getparint("n3",&sim->n3)) sim->n3=1;//default, 2D
  if(sim->n3<1) err("n3 must be positive");
  if(sim->n3>1) {//ny>1, 3D
    if(!getparfloat("d3",&sim->d3)) err("must give d3= for FD grid");
    if(sim->d3<=0) err("d3 must be positive");
    sim->n3pad = sim->n3+2*sim->nb;
    sim->volume = sim->d1*sim->d2*sim->d3;
  }else{
    sim->d3 = 1;
    sim->n3pad = 1;
    sim->volume = sim->d1*sim->d2;
  }
  if(!getparint("order",&sim->order)) sim->order = 4;//only accepts 4 or 8-th order FD
  if(sim->order!=4 && sim->order!=8) err("order must be 4 or 8");
  sim->ri = sim->order/2;//interpolation radius of Bessel I0 function for sinc
  if(sim->nb<sim->ri) err("nb must be at least order/2");
  sim->n123 = sim->n1*sim->n2*sim->n3;
  sim->n123pad = sim->n1pad*sim->n2pad*sim->n3pad;
  
  if(!getparint("eachopt", &sim->eachopt)) sim->eachopt = 0;//1=each shot use different source wavelet
  if(!getparint("freesurf", &sim->freesurf)) sim->freesurf = 1;// 1=free surface; 0=no freesurf
  if(!getparfloat("freq",&sim->freq)) sim->freq = 15;//reference frequency for PML
  if(sim->eachopt!=0 && sim->eachopt!=1) err("eachopt must be 0 or 1");
  if(sim->freesurf!=0 && sim->freesurf!=1) err("freesurf must be 0 or 1");
  if(sim->freq<=0) err("freq must be positive");
  
  /* Three standard-linear-solid mechanisms are fitted to a constant-Q
   * target. nrelax=0 remains the exact acoustic compatibility mode. */
  if(!getparint("nrelax", &sim->nrelax)) sim->nrelax = 0;
  if(sim->nrelax!=0 && sim->nrelax!=3)
    err("nrelax must be 0 (acoustic) or 3 (three SLS mechanisms)");
  {
    float q0, sls_fmin, sls_fmax;
    int il;
    for(il=0; il<3; il++){
      sim->sls_wl[il] = 0.0f;
      sim->sls_yl[il] = 0.0f;
      sim->sls_exp[il] = 1.0f;
      sim->sls_1_exp[il] = 0.0f;
    }
    if(sim->nrelax==3){
      if(!getparfloat("Q0", &q0)) q0 = 100.0f;
      if(!getparfloat("sls_fmin", &sls_fmin)) sls_fmin = 1.0f;//minimum frequency of SLS
      if(!getparfloat("sls_fmax", &sls_fmax)) sls_fmax = 40.0f;//maximum frequency of SLS
      if(q0<=0.0f) err("Q0 must be positive");
      if(sls_fmin<=0.0f || sls_fmax<sls_fmin)
        err("sls_fmin and sls_fmax must satisfy 0 < sls_fmin <= sls_fmax");
      visco_sls_fit(3, sls_fmin, sls_fmax, q0, sim->sls_wl, sim->sls_yl);
      for(il=0; il<3; il++){
        sim->sls_exp[il] = expf(-sim->sls_wl[il]*sim->dt);
        sim->sls_1_exp[il] = 1.0f - sim->sls_exp[il];
      }
    }
  }
  if(iproc==0){
    printf("nt=%d (number of time steps)\n", sim->nt);
    printf("nsnap=%d (Tapenade binomial checkpoint budget)\n", sim->nsnap);
    printf("dt=%g (time step)\n", sim->dt);
    printf("nb=%d (number of boundary layers)\n", sim->nb);
    printf("[d1, d2, d3]=[%g, %g, %g]\n", sim->d1, sim->d2, sim->d3);
    printf("[n1, n2, n3]=[%d, %d, %d]\n", sim->n1, sim->n2, sim->n3);
    printf("[n1pad, n2pad, n3pad]=[%d, %d, %d]\n", sim->n1pad, sim->n2pad, sim->n3pad);
    printf("order=%d (FD order=4 or 8)\n", sim->order);
    printf("ri=%d (interpolation radius ri=order/4)\n", sim->ri);
    printf("eachopt=%d (1=one source per shot; 0=one source for all shots)\n", sim->eachopt);
    printf("freesurf=%d (1=with free surface; 0=no free surface)\n", sim->freesurf);
    printf("nrelax=%d (0=acoustic; 3=visco-acoustic with 3 SLS mechanisms)\n", sim->nrelax);
    if(sim->nrelax==3){
      int il;
      for(il=0; il<3; il++)
        printf("SLS[%d]: omega=%g, y=%g, decay=%g\n", il+1, sim->sls_wl[il], sim->sls_yl[il], sim->sls_exp[il]);
    }
  }

  //=================== specify acquisition ================
  if(!getparint("suopt", &acq->suopt)) acq->suopt = 0;//0=default, 1=for real data precessing using RTM,FWI,LSRTM
  if(acq->suopt!=0 && acq->suopt!=1) err("suopt must be 0 or 1");
  if(!getparfloat("zmin", &acq->zmin)) acq->zmin = 0;
  if(!getparfloat("zmax", &acq->zmax)) acq->zmax = acq->zmin+(sim->n1-1)*sim->d1;
  if(!getparfloat("xmin", &acq->xmin)) acq->xmin = 0;
  if(!getparfloat("xmax", &acq->xmax)) acq->xmax = acq->xmin+(sim->n2-1)*sim->d2;
  if(!getparfloat("ymin", &acq->ymin)) acq->ymin = 0;
  if(!getparfloat("ymax", &acq->ymax)) acq->ymax = acq->ymin+(sim->n3-1)*sim->d3;
  if(acq->zmax<acq->zmin || acq->xmax<acq->xmin || acq->ymax<acq->ymin)
    err("model coordinate maximum must not be less than its minimum");
  if(iproc==0){
    printf("-------- input model range -----------\n");
    printf("[zmin, zmax]=[%g, %g]\n", acq->zmin, acq->zmax);
    printf("[xmin, xmax]=[%g, %g]\n", acq->xmin, acq->xmax);
    if(sim->n3>1) printf("[ymin, ymax]=[%g, %g]\n", acq->ymin, acq->ymax);
  }
  acq->shot_idx = alloc1int(nproc);
  int nsrc = countparval("shots");
  if(nsrc>0){
    if(nsrc!=nproc) err("shots length (%d) must equal nproc (%d)", nsrc, nproc);
    getparint("shots", acq->shot_idx);/* a list of source index separated by comma */
  }
  if(nsrc==0){
    for(int j=0; j<nproc; j++) acq->shot_idx[j] = j+1;//index starts from 1
  }
  
  if(!acq->suopt) acq_init(sim, acq);//read acquisition file if suopt==0
  if(sim->mode>=1 && sim->mode<=7){//mode=1,2,3,4,5,6,7,9 requires reading data
    read_data(sim, acq);//read data in binary or SU format
    setup_data_mask(acq, sim);//the muting will be used to remove direct waves
  }
  ierr = MPI_Barrier(MPI_COMM_WORLD);

  //=================== read vp,rho,qinv,stf files =================
  if(!getparstring("vpfile",&vpfile)) err("must give vpfile= ");
  sim->vp = alloc3float(sim->n1, sim->n2, sim->n3);
  fp=fopen(vpfile, "rb");
  if(fp==NULL) err("cannot open vpfile=%s", vpfile);
  if(fread(&sim->vp[0][0][0], sizeof(float), sim->n123, fp)!=(size_t)sim->n123)
    err("error reading vpfile=%s,  size unmatched", vpfile);
  fclose(fp);
  
  if(!getparstring("rhofile",&rhofile)) err("must give rhofile= ");
  sim->rho = alloc3float(sim->n1, sim->n2, sim->n3);
  fp = fopen(rhofile, "rb");
  if(fp==NULL) err("cannot open rhofile=%s", rhofile);
  if(fread(&sim->rho[0][0][0], sizeof(float), sim->n123, fp)!=(size_t)sim->n123)
    err("error reading rhofile=%s,  size unmatched", rhofile);
  fclose(fp);

  sim->qinv = alloc3float(sim->n1, sim->n2, sim->n3);
  if(sim->nrelax>0){
    if(!getparstring("qinvfile", &qinvfile))
      err("must give qinvfile= when nrelax>0");
    fp = fopen(qinvfile, "rb");
    if(fp==NULL) err("cannot open qinvfile=%s", qinvfile);
    if(fread(&sim->qinv[0][0][0], sizeof(float), sim->n123, fp)!=(size_t)sim->n123)
      err("error reading qinvfile=%s,  size unmatched", qinvfile);
    fclose(fp);
  }else{
    memset(sim->qinv[0][0], 0, sim->n123*sizeof(float));
  }
  
  sim->stf = alloc1float(sim->nt); //source wavelet
  if(sim->mode!=5){
    if(!getparstring("stffile",&stffile)) err("must give stffile= ");
    if(sim->eachopt){//read each source wavelet for every shot
      char number[sizeof("0000")];
      char fname[PATH_MAX];
      sprintf(number, "%04d", acq->shot_idx[iproc]);
      //sources will be named stf_0001, stf_0002, ..., where stffile='stf'
      snprintf(fname, sizeof(fname), "%s_%s", stffile, number);
    
      fp=fopen(fname,"rb");
      if(fp==NULL) err("cannot open stffile=%s", fname);
      if(fread(sim->stf, sizeof(float), sim->nt, fp)!=(size_t)sim->nt) 
	err("error reading stffile=%s,  size unmatched", fname);
      fclose(fp);
    }else{//read the same wavelet for all shots
      fp=fopen(stffile, "rb");
      if(fp==NULL) err("cannot open stffile=%s", stffile);
      if(fread(sim->stf, sizeof(float), sim->nt, fp)!=(size_t)sim->nt) 
	err("error reading stffile=%s,  size unmatched", stffile);
      fclose(fp);
    }
  }

  //====================do the job here========================
  if(sim->mode==0) do_modelling(sim, acq);
  else if(sim->mode==1) do_fwi(sim, acq);
  else if(sim->mode==2) do_rtm(sim, acq);
  else if(sim->mode==3) do_lsm(sim, acq);
  else if(sim->mode==4) do_fwi(sim, acq);
  else if(sim->mode==5) do_invert_source(sim, acq);
  
  //===========================================================
  free(sim->stf);
  free3float(sim->vp);
  free3float(sim->rho);
  free3float(sim->qinv);

  acq_free(sim, acq);  
  free(sim);
  free(acq);
  
  MPI_Finalize();

  return 0;
}
