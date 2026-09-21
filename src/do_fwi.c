/* nonlinear optimization for full waveform inversion (FWI)
 *---------------------------------------------------------------------------
 *  Copyright (c) Pengliang Yang, 2026, Laoshan Laboratory, China
 *  Copyright (c) Pengliang Yang, 2020, Harbin Institute of Technology, China
 *  Copyright (c) Pengliang Yang, 2018, University Grenoble Alpes, France
 *  Homepage: https://yangpl.wordpress.com
 *  E-mail: ypl.2100@gmail.com
 *--------------------------------------------------------------------------*/
#include <mpi.h>
#include "cstd.h"
#include "acq.h"
#include "opt.h"
#include "sim.h"
#include "fwi.h"

fwi_t *fwi;

float l2norm(int n, float *a);
void flipsign(int n, float *a, float *b);
void lbfgs_save(int n, float *x, float *g, float **sk, float **yk, opt_t *opt);
void lbfgs_update(int n, float *x, float *g, float **sk, float **yk, opt_t *opt);
void lbfgs_descent(int n, float *g, float *d, float **sk, float **yk, opt_t *opt);
int lbfgs_descent1(int n, float *g, float *q, float *rho, float *alp, float **sk, float **yk, opt_t *opt);
void lbfgs_descent2(int n, float *q, float *rho, float *alp, float **sk, float **yk, opt_t *opt);
void boundx(float *x, int n, float *xmin, float *xmax);
void line_search(int n, //dimension of x
		 float *x, //input vector x
		 float *g, //gradient of misfit function
		 float *d, //descent direction
		 opt_fg fg, //subroutine to evaluation function and gradient
		 opt_t *opt); //pointer of l-BFGS optimization parameters
void cg_solve(int n, //dimension of x
	      float *x, //input vector x
	      float *g, //gradient of misfit function
	      float *d, //descent direction
	      opt_Hv Hv, //Hessian-vector product
	      opt_precondition P, //optional SPD preconditioner
	      opt_t *opt); //pointer of l-BFGS optimization parameters


double fg_fwi_eval(float *x, float *g);
float fg_fwi(float *x, float *g);
void gradient_check(sim_t *sim, fwi_t *fwi, float *x, float *g, double fcost,
                    int checkpar, float h, float h2);
void precondition(sim_t *sim, fwi_t *fwi, float *x);
void Hv_fwi(float *x, float *v, float *Hv);
void write_vector(const char *name, const float *x, int n);
void append_vector(const char *name, const float *x, int n);

extern sim_t *sim;

static void fwi_precondition(float *x)
{
  precondition(sim, fwi, x);
}

void do_fwi(sim_t *sim, acq_t *acq)
/*< perform FWI using l-BFGS optimization >*/
{
  int i1, i2, i3, ipar, j;
  int checkgrad, checkpar;
  float checkh, checkh2, beta_nlcg, ggprev;
  double fcost;
  opt_t *opt; //pointer for opt_t parameters
  char *bathyfile;
  FILE *fp;
  float *gprev = NULL, *dprev = NULL, *zprev = NULL;

  opt = malloc(sizeof(opt_t));
  if(opt==NULL) err("cannot allocate optimizer state");
  if(!getparint("niter", &opt->niter)) opt->niter=50;//maximum number of iterations
  if(!getparint("nls", &opt->nls)) opt->nls=20;//maximum number of line searches
  if(!getparfloat("tol", &opt->tol)) opt->tol=1e-8;//convergence tolerance 
  if(!getparint("npair", &opt->npair)) opt->npair=5; //l-BFGS memory length
  if(!getparfloat("c1", &opt->c1)) opt->c1=1e-4; //Nocedal value for Wolfe condition
  if(!getparfloat("c2", &opt->c2)) opt->c2=0.9;  //Nocedal value for Wolfe condition
  if(!getparint("bound", &opt->bound)) opt->bound=1;//use bounds or not
  if(!getparint("preco", &opt->preco)) opt->preco=1;
  if(opt->preco!=0 && opt->preco!=1) err("preco must be 0 or 1");
  if(!getparfloat("alpha", &opt->alpha)) opt->alpha=1.0;//alpha
  if(!getparint("opt_method", &opt->opt_method)) opt->opt_method=1;//0=NLCG, 1=l-BFGS; 2=Newton-CG
  if(opt->opt_method<0 || opt->opt_method>2)
    err("opt_method must be 0 (NLCG), 1 (l-BFGS), or 2 (Newton-CG)");
  if(!getparint("ncg", &opt->ncg)) opt->ncg=10;
  if(opt->ncg<1) err("ncg must be positive");
  if(!getparfloat("tol_cg", &opt->tol_cg)) opt->tol_cg=1.0e-2f;
  if(!(opt->tol_cg>0.0f && opt->tol_cg<1.0f))
    err("tol_cg must be in (0,1)");
  opt->verb = (iproc==0)?1:0; //other process are silent.

  fwi = malloc(sizeof(fwi_t));
  if(fwi==NULL) err("cannot allocate FWI state");
  fwi->bathy=alloc2float(sim->n2, sim->n3);
  fwi->ibathy=alloc2int(sim->n2, sim->n3);
  if(!getparstring("bathyfile",&bathyfile)){
    memset(fwi->bathy[0], 0, sim->n2*sim->n3*sizeof(float));
    memset(fwi->ibathy[0], 0, sim->n2*sim->n3*sizeof(int));
  }else{
    fp=fopen(bathyfile,"rb");
    if(fp==NULL) err("cannot open bathyfile=%s",bathyfile);
    if(fread(fwi->bathy[0],sizeof(float),sim->n2*sim->n3,fp)!=(size_t)(sim->n2*sim->n3)) 
      err("error reading bathyfile=%s", bathyfile);
    fclose(fp);
    for(i3=0; i3<sim->n3; i3++){
      for(i2=0; i2<sim->n2; i2++){
	fwi->ibathy[i3][i2]=NINT(fwi->bathy[i3][i2]/sim->d1);
      }
    }
  }
  if(sim->mode==2){
    fwi->family = 2;
    fwi->npar = 1;
  }else{
    if(!getparint("family", &fwi->family)) fwi->family = 1;//1=vp-rho'; 2=vp-Ip
    if(!getparint("npar", &fwi->npar)) fwi->npar = 1;//number of unknown parameters
  }
  if(fwi->family!=1 && fwi->family!=2)
    err("family must be 1 (vp-rho) or 2 (vp-Ip)");
  if(fwi->npar<1 || fwi->npar>3)
    err("npar must be in [1,3]");
  fwi->idxpar = alloc1int(fwi->npar);
  if(sim->mode==2){
    fwi->idxpar[0] = 2;
  }else{
    if(!(j=countparval("idxpar"))) err("must give idxpar= vector");
    if(j!= fwi->npar) err("must have length[idxpar]=%d", fwi->npar);
    getparint("idxpar", fwi->idxpar);
  }
  //family=1: idxpar=1, vp; idxpar=2, rho; family=2: idxpar=1, vp; idxpar=2, Ip.
  //idxpar=3 selects log inverse-Q and requires the viscoacoustic path.
  for(ipar=0; ipar<fwi->npar; ipar++){
    if(fwi->idxpar[ipar]<1 || fwi->idxpar[ipar]>3)
      err("idxpar values must be 1 (vp), 2 (rho/Ip), or 3 (qinv)");
    if(fwi->idxpar[ipar]==3 && sim->nrelax==0)
      err("idxpar=3 (qinv) requires nrelax=3");
  }
  if(!getparfloat("rhomin", &fwi->rhomin)) fwi->rhomin = 1000;
  if(!getparfloat("rhomax", &fwi->rhomax)) fwi->rhomax = 3000;
  if(!getparfloat("vpmin", &fwi->vpmin)) fwi->vpmin = 1000;
  if(!getparfloat("vpmax", &fwi->vpmax)) fwi->vpmax = 6000;
  if(!getparfloat("qinvmin", &fwi->qinvmin)) fwi->qinvmin = 1e-6f;
  if(!getparfloat("qinvmax", &fwi->qinvmax)) fwi->qinvmax = 1.0f;
  if(!(fwi->vpmin>0.0f) || !(fwi->vpmax>=fwi->vpmin))
    err("vpmin and vpmax must satisfy 0 < vpmin <= vpmax");
  if(!(fwi->rhomin>0.0f) || !(fwi->rhomax>=fwi->rhomin))
    err("rhomin and rhomax must satisfy 0 < rhomin <= rhomax");
  if(!(fwi->qinvmin>0.0f) || !(fwi->qinvmax>=fwi->qinvmin))
    err("qinvmin and qinvmax must satisfy 0 < qinvmin <= qinvmax");
  
  if(iproc==0) {
    printf("family=%d (1=vp-rho; 2=vp-ip)\n", fwi->family);
    printf("npar=%d\n", fwi->npar);
    printf("parameter bound: [rhomin, rhomax]=[%g,%g]\n", fwi->rhomin, fwi->rhomax);
    printf("parameter bound: [vpmin, vpmax]=[%g,%g]\n", fwi->vpmin, fwi->vpmax);
    printf("parameter bound: [qinvmin, qinvmax]=[%g,%g]\n", fwi->qinvmin, fwi->qinvmax);
  }
  
  fwi->n = sim->n123*fwi->npar;
  opt->x = alloc1float(fwi->n);
  opt->g = alloc1float(fwi->n);
  opt->d = alloc1float(fwi->n);
  if(opt->preco) opt->pg = alloc1float(fwi->n);
  opt->sk= alloc2float(fwi->n, opt->npair);
  opt->yk= alloc2float(fwi->n, opt->npair);
  if(opt->opt_method==1){
    gprev = alloc1float(fwi->n);
    dprev = alloc1float(fwi->n);
    zprev = alloc1float(fwi->n);
  }
  if(opt->bound){
    opt->xmin = alloc1float(fwi->n);
    opt->xmax = alloc1float(fwi->n);
    /* Optimization happens in transformed variables, mostly log-parameters,
     * so the box constraints must be defined in that same space. */
    for(ipar=0; ipar<fwi->npar; ipar++){
      for(j=0; j<sim->n123; j++){
	if(fwi->family==1){//1=vp; 2=rho
	  if(fwi->idxpar[ipar]==1) {
	    opt->xmin[j+ipar*sim->n123] = log(fwi->vpmin);
	    opt->xmax[j+ipar*sim->n123] = log(fwi->vpmax);
	  }
	  if(fwi->idxpar[ipar]==2) {
	    opt->xmin[j+ipar*sim->n123] = log(fwi->rhomin);
	    opt->xmax[j+ipar*sim->n123] = log(fwi->rhomax);
	  }
	}else if(fwi->family==2){//1=vp; 2=ip
	  if(fwi->idxpar[ipar]==1) {
	    opt->xmin[j+ipar*sim->n123] = log(fwi->vpmin);
	    opt->xmax[j+ipar*sim->n123] = log(fwi->vpmax);
	  }
	  if(fwi->idxpar[ipar]==2) {
	    opt->xmin[j+ipar*sim->n123] = log(fwi->rhomin*fwi->vpmin);
	    opt->xmax[j+ipar*sim->n123] = log(fwi->rhomax*fwi->vpmax);
	  }
	}//end if
	if(fwi->idxpar[ipar]==3) {
	  opt->xmin[j+ipar*sim->n123] = log(fwi->qinvmin);
	  opt->xmax[j+ipar*sim->n123] = log(fwi->qinvmax);
	}
      }//end for j
    }//end for ipar
  }//end if
  
  if(iproc==0) printf("------------- fwi init ----------------\n");
  if(!getparint("itcheck", &sim->itcheck)) sim->itcheck = sim->nt/2;

  fwi->iter = 0;
  fwi->alpha = 1;
  fwi->firstgrad = 1;
  if(!getparint("objopt", &fwi->objopt)) fwi->objopt = 0;
  if(fwi->objopt!=0)
    err("Tapenade FWI currently supports only objopt=0 (weighted L2)");

  fwi->niter = opt->niter;

  if(!getparint("checkgrad", &checkgrad)) checkgrad = 0;
  if(!getparint("checkpar", &checkpar)) checkpar = 0;
  if(!getparfloat("checkh", &checkh)) checkh = 5e-3f;
  if(!getparfloat("checkh2", &checkh2)) checkh2 = 0.5f*checkh;
  if(checkgrad && sim->mode!=1 && sim->mode!=4)
    err("checkgrad requires mode=1 or mode=4");
  if(checkgrad && (checkpar<0 || checkpar>3))
    err("checkpar must be 0 (all), 1 (vp), 2 (rho/Ip), or 3 (qinv)");
  if(checkgrad && (!(checkh>0.0f) || !(checkh2>0.0f)))
    err("checkh and checkh2 must be positive");
	
  if(sim->mode==1||sim->mode==2||sim->mode==4) {//FWI, RTM, or FWI gradient
    /* Each inversion parameter occupies one contiguous n123 slab in `opt->x`.
     * The same flattened ordering is used throughout optimization and I/O. */
    //initialize opt->x[]
    for(ipar=0; ipar<fwi->npar; ipar++){
      for(i3=0; i3<sim->n3; i3++){
	for(i2=0; i2<sim->n2; i2++){
	  for(i1=0; i1<sim->n1; i1++){
	    j = i1 + sim->n1*(i2 + sim->n2*(i3 + sim->n3*ipar));
	    if(fwi->family==1){//vp-rho
	      if(fwi->idxpar[ipar]==1) opt->x[j] = log(sim->vp[i3][i2][i1]);
	      if(fwi->idxpar[ipar]==2) opt->x[j] = log(sim->rho[i3][i2][i1]);
	    }
	    if(fwi->family==2){//vp-ip
	      if(fwi->idxpar[ipar]==1) opt->x[j] = log(sim->vp[i3][i2][i1]);
	      if(fwi->idxpar[ipar]==2) opt->x[j] = log(sim->rho[i3][i2][i1]*sim->vp[i3][i2][i1]);
	    }
	    if(fwi->idxpar[ipar]==3){
	      if(!(sim->qinv[i3][i2][i1]>0.0f))
	        err("qinv must be positive when idxpar=3");
	      opt->x[j] = log(sim->qinv[i3][i2][i1]);
	    }
	  }//end for i1
	}//end for i2
      }//end for i3
    }//end for ipar
    fcost = fg_fwi_eval(opt->x, opt->g);
    /* Optional validation only; the normal FWI/gradient path is unchanged. */
    if(checkgrad)
      gradient_check(sim, fwi, opt->x, opt->g, fcost,
                     checkpar, checkh, checkh2);
  }

  if(sim->mode==1){//FWI
    //initialize all counters
    opt->f0 = (float)fcost;
    opt->fk = (float)fcost;
    if(!(opt->alpha > 0.)) opt->alpha = 1.;
    opt->igrad = 0;
    opt->kpair = 0;
    opt->ils = 0;
    opt->ls_fail = 0;
    if(opt->verb){
      opt->gk_norm = l2norm(fwi->n, opt->g);
      fp=fopen("iterate.txt","w");
      if(fp==NULL) err("cannot open iterate.txt for writing");
      fprintf(fp,"==========================================================\n");
      fprintf(fp,"opt_method=%d (0=NLCG; 1=L-BFGS; 2=Newton-CG)\n",opt->opt_method);
      fprintf(fp,"Maximum number of iterations: %d\n",opt->niter);
      fprintf(fp,"Convergence tolerance: %3.2e\n", opt->tol);
      fprintf(fp,"maximum number of line search: %d\n",opt->nls);
      fprintf(fp,"initial step length: alpha=%g\n",opt->alpha);
      fprintf(fp,"precodition=%d (1=preco; 0=not)\n", opt->preco);
      if(opt->opt_method==1){
      	fprintf(fp,"l-BFGS memory length: %d\n",opt->npair);
      }else if(opt->opt_method==2){
      	fprintf(fp,"ncg=%d\n", opt->ncg);
      	fprintf(fp,"tol_cg=%g\n", opt->tol_cg);
      }
      fprintf(fp,"==========================================================\n");
      fprintf(fp,"iter    fk       fk/f0      ||gk||    alpha    nls   ngrad\n");
      fclose(fp);
    }
    //l-BFGS optimization 
    for(fwi->iter=0; fwi->iter<opt->niter; fwi->iter++){
      if(opt->verb){
	printf("==========================================================\n");
	printf("# iter=%d  fk/f0=%g\n", fwi->iter,opt->fk/opt->f0);
	opt->gk_norm = l2norm(fwi->n, opt->g);
	fp=fopen("iterate.txt","a");
	if(fp==NULL) err("cannot open iterate.txt for appending");
	fprintf(fp,"%3d   %3.2e  %3.2e   %3.2e  %3.2e  %3d  %4d\n",
		fwi->iter,opt->fk,opt->fk/opt->f0,opt->gk_norm,opt->alpha,opt->ils,opt->igrad);
	fclose(fp);
      }
      if(opt->opt_method==0){
	/* Polak-Ribiere NLCG restarts automatically when beta becomes negative. */
	float *z = opt->g;
	if(opt->preco){
	  memcpy(opt->pg, opt->g, fwi->n*sizeof(float));
	  precondition(sim, fwi, opt->pg);
	  z = opt->pg;
	}
	if(fwi->iter==0){
	  flipsign(fwi->n, z, opt->d);
	}else{
	  beta_nlcg = 0.0f;
	  ggprev = 0.0f;
	  for(j=0; j<fwi->n; j++){
	    beta_nlcg += opt->g[j]*(z[j]-zprev[j]);
	    ggprev += gprev[j]*zprev[j];
	  }
	  beta_nlcg = ggprev>0.0f ? MAX(0.0f, beta_nlcg/ggprev) : 0.0f;
	  for(j=0; j<fwi->n; j++)
	    opt->d[j] = -z[j]+beta_nlcg*dprev[j];
	}
	memcpy(gprev, opt->g, fwi->n*sizeof(float));
	memcpy(zprev, z, fwi->n*sizeof(float));
	memcpy(dprev, opt->d, fwi->n*sizeof(float));
      }else if(opt->opt_method==1){
	if(fwi->iter==0){
	  if(opt->preco){
	    memcpy(opt->pg, opt->g, fwi->n*sizeof(float));
	    precondition(sim, fwi, opt->pg);
	    flipsign(fwi->n, opt->pg, opt->d);
	  }else
	    flipsign(fwi->n, opt->g, opt->d);
	}else{
	  opt->q = alloc1float(fwi->n);
	  opt->rho = alloc1float(opt->kpair);
	  opt->alp = alloc1float(opt->kpair);
	  lbfgs_update(fwi->n, opt->x, opt->g, opt->sk, opt->yk, opt);
	  opt->loop1=lbfgs_descent1(fwi->n, opt->g, opt->q, opt->rho,
	                             opt->alp, opt->sk, opt->yk, opt);
	  if(opt->preco) precondition(sim, fwi, opt->q);
	  if(opt->loop1) lbfgs_descent2(fwi->n, opt->q, opt->rho, opt->alp,
	                                opt->sk, opt->yk, opt);
	  flipsign(fwi->n, opt->q, opt->d);
	  free1float(opt->q);
	  free1float(opt->alp);
	  free1float(opt->rho);
	}
	lbfgs_save(fwi->n, opt->x, opt->g, opt->sk, opt->yk, opt);
      }else{
	/* Newton-CG solves H_GN d=-g with the TLM/adjoint Gauss-Newton product. */
	cg_solve(fwi->n, opt->x, opt->g, opt->d, Hv_fwi,
	         opt->preco ? fwi_precondition : NULL, opt);
      }
      ggprev = 0.0f;
      for(j=0; j<fwi->n; j++) ggprev += opt->g[j]*opt->d[j];
      if(ggprev>=0.0f){
	if(opt->opt_method==1 && opt->preco)
	  flipsign(fwi->n, opt->pg, opt->d);
	else
	  flipsign(fwi->n, opt->g, opt->d);
      }
      line_search(fwi->n, opt->x, opt->g, opt->d, fg_fwi, opt);
      
      if(opt->ls_fail){
	if(opt->verb) {
	  fp=fopen("iterate.txt","a");
	  if(fp==NULL) err("cannot open iterate.txt for appending");
	  fprintf(fp, "==>Line search failed!\n");
	  fclose(fp);
	}
	break;
      }
      //not break, then line search succeeds or descent direction accepted
      if(opt->verb) {
	float *param = alloc1float(fwi->n);
	for(j=0; j<fwi->n; j++){
	  param[j] = exp(opt->x[j]);
	}
	write_vector("param_final", param, fwi->n);
	write_vector("gradient_final", opt->g, fwi->n);
	
	if(sim->n3==1){//we only store intermediate models and gradients in 2D 
	  if(fwi->iter==0){
	    write_vector("param_iter", param, fwi->n);
	    write_vector("gradient_iter", opt->g, fwi->n);
	  }else{
	    append_vector("param_iter", param, fwi->n);
	    append_vector("gradient_iter", opt->g, fwi->n);
	  }
	}//end if n3>1
	free1float(param);
      }

      if(opt->fk < opt->tol * opt->f0){//here we assume misfit function is always positive
	if(opt->verb){
	  fp=fopen("iterate.txt","a");
	  if(fp==NULL) err("cannot open iterate.txt for appending");
	  fprintf(fp, "==>Convergence reached!\n");
	  fclose(fp);
	}
	break;
      }
      fflush(stdout);
    } 
    if(opt->verb && fwi->iter==opt->niter) {
      fp=fopen("iterate.txt","a");
      if(fp==NULL) err("cannot open iterate.txt for appending");
      fprintf(fp, "==>Maximum iteration number reached!\n");
      fclose(fp);
   }
  }

  if(opt->opt_method==1){
    free1float(zprev);
    free1float(dprev);
    free1float(gprev);
  }
  free1int(fwi->idxpar);
  free2float(fwi->bathy);
  free2int(fwi->ibathy);
  free(fwi);
  
  free1float(opt->x);
  free1float(opt->g);
  free1float(opt->d);
  if(opt->preco) free1float(opt->pg);
  free2float(opt->sk);
  free2float(opt->yk);
  if(opt->bound){
    free1float(opt->xmin);
    free1float(opt->xmax);
  }
  free(opt);
}
