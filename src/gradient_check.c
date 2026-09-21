/* Centered finite-difference checks for the Tapenade FWI gradient. */
#include "cstd.h"
#include "sim.h"
#include "fwi.h"

extern int iproc;

double fg_fwi_eval(float *x, float *g);

/*
 * Build one reproducible direction in one inversion-parameter slab.  The
 * bathymetry mask is excluded because the FWI gradient deliberately returns zero
 * gradient there, while those model samples still participate in modelling.
 */
static int make_direction(const sim_t *sim, const fwi_t *fwi, int ipar,
                          float *direction)
{
  int i1, i2, i3, j, active;
  unsigned int state;

  memset(direction, 0, (size_t)fwi->n*sizeof(float));
  state = 0x9e3779b9u ^ (unsigned int)(ipar+1)*0x85ebca6bu;
  active = 0;
  for(i3=0; i3<sim->n3; i3++)
    for(i2=0; i2<sim->n2; i2++)
      for(i1=0; i1<sim->n1; i1++){
        if(i1<=fwi->ibathy[i3][i2]) continue;
        j = i1 + sim->n1*(i2 + sim->n2*(i3 + sim->n3*ipar));
        /* Numerical Recipes' LCG is sufficient here and avoids global rand(). */
        state = 1664525u*state + 1013904223u;
        direction[j] = (state & 0x80000000u) ? 1.0f : -1.0f;
        active++;
      }
  return active;
}

static void check_one_step(const fwi_t *fwi, const float *x,
                           const float *base_gradient, const float *direction,
                           float *xplus, float *xminus,
                           float *gplus, float *gminus, float h,
                           double *fd_gradient, double *adjoint_gradient,
                           double *jplus, double *jminus)
{
  int j;
  double dot = 0.0;

  memcpy(xplus, x, (size_t)fwi->n*sizeof(float));
  memcpy(xminus, x, (size_t)fwi->n*sizeof(float));
  for(j=0; j<fwi->n; j++){
    xplus[j] += h*direction[j];
    xminus[j] -= h*direction[j];
    dot += (double)base_gradient[j]*(double)direction[j];
  }

  *jplus = fg_fwi_eval(xplus, gplus);
  *jminus = fg_fwi_eval(xminus, gminus);
  *fd_gradient = (*jplus-*jminus)/(2.0*(double)h);
  *adjoint_gradient = dot;
}

void gradient_check(sim_t *sim, fwi_t *fwi, float *x, float *g, double fcost,
                    int checkpar, float h, float h2)
/*< run a second-order centered finite-difference directional check >*/
{
  int ipar, selected, active, nselected;
  float *base_gradient, *direction, *xplus, *xminus;
  float *gplus, *gminus, *grestore;
  int saved_firstgrad;
  float saved_alpha;
  double saved_fcost;

  if(checkpar<0 || checkpar>3)
    err("checkpar must be 0 (all), 1 (vp), 2 (rho/Ip), or 3 (qinv)");
  if(!(h>0.0f) || !(h2>0.0f))
    err("checkh and checkh2 must be positive");

  nselected = 0;
  for(ipar=0; ipar<fwi->npar; ipar++){
    selected = checkpar==0 || fwi->idxpar[ipar]==checkpar;
    if(selected) nselected++;
  }
  if(nselected==0)
    err("checkpar=%d is not present in idxpar", checkpar);

  base_gradient = alloc1float(fwi->n);
  direction = alloc1float(fwi->n);
  xplus = alloc1float(fwi->n);
  xminus = alloc1float(fwi->n);
  gplus = alloc1float(fwi->n);
  gminus = alloc1float(fwi->n);
  grestore = alloc1float(fwi->n);
  memcpy(base_gradient, g, (size_t)fwi->n*sizeof(float));

  saved_firstgrad = fwi->firstgrad;
  saved_alpha = fwi->alpha;
  saved_fcost = fwi->fcost;
  /* Keep the scale selected by the base evaluation fixed for every trial. */
  fwi->firstgrad = 0;
  if(iproc==0){
    printf("---- second-order centered gradient check ----\n");
    printf("direction: deterministic +/-1 over active cells (log-parameter units)\n");
    printf("base objective=%g, h=%g, h2=%g\n", fcost, h, h2);
  }

  for(ipar=0; ipar<fwi->npar; ipar++){
    if(checkpar!=0 && fwi->idxpar[ipar]!=checkpar) continue;
    active = make_direction(sim, fwi, ipar, direction);
    if(active==0)
      err("no active cells remain for gradient check slab %d", ipar);

    {
      double fd1, fd2, adj1, adj2, jp1, jm1, jp2, jm2;
      double rel1, rel2, ratio;

      check_one_step(fwi, x, base_gradient, direction,
                     xplus, xminus, gplus, gminus, h,
                     &fd1, &adj1, &jp1, &jm1);
      check_one_step(fwi, x, base_gradient, direction,
                     xplus, xminus, gplus, gminus, h2,
                     &fd2, &adj2, &jp2, &jm2);
      {
        /* relative error = |a-b| / max(|a|,|b|) (was relative_error()) */
        double denom1 = MAX(fabs(fd1), fabs(adj1));
        double denom2 = MAX(fabs(fd2), fabs(adj2));
        rel1 = denom1>0.0 ? fabs(fd1-adj1)/denom1 : fabs(fd1-adj1);
        rel2 = denom2>0.0 ? fabs(fd2-adj2)/denom2 : fabs(fd2-adj2);
      }
      ratio = rel2>0.0 ? rel1/rel2 : 0.0;

      /* parameter label (was gradient_parameter_name()) */
      const char *parname;
      int parameter = fwi->idxpar[ipar];
      if(parameter==1) parname = "ln(vp)";
      else if(parameter==3) parname = "ln(qinv)";
      else if(fwi->family==1) parname = "ln(rho)";
      else parname = "ln(Ip)";
      if(iproc==0){
        printf("check slab=%d parameter=%s active=%d\n",
               ipar, parname, active);
        printf("  h=%g:   Jplus=%g Jminus=%g FD=%g adjoint=%g relerr=%g\n",
               h, jp1, jm1, fd1, adj1, rel1);
        printf("  h2=%g:  Jplus=%g Jminus=%g FD=%g adjoint=%g relerr=%g\n",
               h2, jp2, jm2, fd2, adj2, rel2);
        printf("  error ratio relerr(h)/relerr(h2)=%g (second-order target: 4)\n",
               ratio);
      }
    }
  }

  /* Restore the physical model and the caller's objective/gradient state. */
  (void)fg_fwi_eval(x, grestore);
  memcpy(g, base_gradient, (size_t)fwi->n*sizeof(float));
  fwi->firstgrad = saved_firstgrad;
  fwi->alpha = saved_alpha;
  fwi->fcost = saved_fcost;

  free1float(grestore);
  free1float(gminus);
  free1float(gplus);
  free1float(xminus);
  free1float(xplus);
  free1float(direction);
  free1float(base_gradient);
}
