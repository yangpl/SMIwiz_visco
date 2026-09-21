/* Constant-Q SLS fit adapted from ../doc/lsq.c.
 *
 * Fit Q^{-1}(w) = sum_l Y_l A_l(w) / (1 - sum_l Y_l B_l(w))
 * with projected, multi-start Levenberg-Marquardt.  Trapezoidal weights
 * on 200 log-spaced nodes approximate the integral over angular frequency.
 * The public interface takes frequencies in Hz and returns angular
 * relaxation frequencies and normalized y_l = Q0*Y_l.  The propagator
 * multiplies y_l by the local inverse quality factor to recover Y_l.
 */
#include <math.h>

#define VISCO_SLS_MAX 3
#define VISCO_SLS_NODES 200
#define VISCO_SLS_MAXIT 200

typedef struct {
  int L;
  double qinv;
  double weight[VISCO_SLS_NODES];
  double A[VISCO_SLS_NODES][VISCO_SLS_MAX];
  double B[VISCO_SLS_NODES][VISCO_SLS_MAX];
} sls_t;

/* Gaussian elimination with partial pivoting, for one to three processes. */
static int sls_solve(int L, double matrix[VISCO_SLS_MAX][VISCO_SLS_MAX],
                     const double *rhs, double *x)
{
  double a[VISCO_SLS_MAX][VISCO_SLS_MAX+1];
  int i, j, k;

  for(i=0; i<L; i++){
    for(j=0; j<L; j++) a[i][j] = matrix[i][j];
    a[i][L] = rhs[i];
  }
  for(k=0; k<L; k++){
    int p = k;
    for(i=k+1; i<L; i++)
      if(fabs(a[i][k]) > fabs(a[p][k])) p = i;
    if(!isfinite(a[p][k]) || a[p][k] == 0.0) return 0;
    if(p != k){
      for(j=0; j<=L; j++){
        double t = a[k][j];
        a[k][j] = a[p][j];
        a[p][j] = t;
      }
    }
    for(i=k+1; i<L; i++){
      double f = a[i][k]/a[k][k];
      for(j=k; j<=L; j++) a[i][j] -= f*a[k][j];
    }
  }
  for(i=L-1; i>=0; i--){
    double s = a[i][L];
    for(j=i+1; j<L; j++) s -= a[i][j]*x[j];
    x[i] = s/a[i][i];
    if(!isfinite(x[i])) return 0;
  }
  return 1;
}

/* Evaluate weighted residuals and, optionally, their analytic Jacobian.
 * A box projection alone does not guarantee a positive denominator;
 * reject singular/nonphysical trial points rather than clamping it.
 */
static double sls_evaluate(const sls_t *sls, const double *Y,
                           double *r, double J[][VISCO_SLS_MAX])
{
  double chi = 0.0;
  int m, l;

  for(m=0; m<VISCO_SLS_NODES; m++){
    double N = 0.0, D = 1.0, Qt, residual, sqrt_weight;
    for(l=0; l<sls->L; l++){
      N += Y[l]*sls->A[m][l];
      D -= Y[l]*sls->B[m][l];
    }
    if(D <= 0.0 || !isfinite(D)) return INFINITY;
    Qt = N/D;
    sqrt_weight = sqrt(sls->weight[m]);
    residual = sqrt_weight*(Qt - sls->qinv);
    chi += residual*residual;
    if(r) r[m] = residual;
    if(J){
      for(l=0; l<sls->L; l++)
        J[m][l] = sqrt_weight*(sls->A[m][l] +
                               Qt*sls->B[m][l])/D;
    }
  }
  return chi;
}

static double sls_lm_solve(const sls_t *sls, double *Y)
{
  double r[VISCO_SLS_NODES];
  double J[VISCO_SLS_NODES][VISCO_SLS_MAX];
  double lambda = 1.0e-3;
  double chi = sls_evaluate(sls, Y, 0, 0);
  int it, m, i, j, trial;

  for(it=0; it<VISCO_SLS_MAXIT; it++){
    double JTJ[VISCO_SLS_MAX][VISCO_SLS_MAX] = {{0}};
    double JTr[VISCO_SLS_MAX] = {0};
    double gnorm = 0.0;
    int accepted = 0;

    chi = sls_evaluate(sls, Y, r, J);
    if(!isfinite(chi)) break;
    for(m=0; m<VISCO_SLS_NODES; m++){
      for(i=0; i<sls->L; i++){
        JTr[i] += J[m][i]*r[m];
        for(j=0; j<sls->L; j++) JTJ[i][j] += J[m][i]*J[m][j];
      }
    }
    for(trial=0; trial<30; trial++){
      double matrix[VISCO_SLS_MAX][VISCO_SLS_MAX];
      double rhs[VISCO_SLS_MAX], dY[VISCO_SLS_MAX];
      double Ynew[VISCO_SLS_MAX], chi_new;

      for(i=0; i<sls->L; i++){
        for(j=0; j<sls->L; j++) matrix[i][j] = JTJ[i][j];
        matrix[i][i] += lambda*(1.0 + JTJ[i][i]);
        rhs[i] = -JTr[i];
      }
      if(sls_solve(sls->L, matrix, rhs, dY)){
        for(i=0; i<sls->L; i++){
          Ynew[i] = Y[i] + dY[i];
          if(Ynew[i] < 0.0) Ynew[i] = 0.0;
          if(Ynew[i] > 1.0) Ynew[i] = 1.0;
        }
        chi_new = sls_evaluate(sls, Ynew, 0, 0);
        if(chi_new < chi){
          for(i=0; i<sls->L; i++) Y[i] = Ynew[i];
          chi = chi_new;
          lambda = fmax(1.0e-12, lambda*0.3);
          accepted = 1;
          break;
        }
      }
      lambda = fmin(1.0e12, lambda*3.0);
    }
    for(i=0; i<sls->L; i++) gnorm += JTr[i]*JTr[i];
    if(!accepted || sqrt(gnorm) < 1.0e-12) break;
  }
  return chi;
}

void visco_sls_fit(int L, float fmin, float fmax, float Q0,
                   float *wl, float *yl)
{
  const double pi = 3.14159265358979323846;
  const double starts[5][VISCO_SLS_MAX] = {
    {1.0/3.0, 1.0/3.0, 1.0/3.0},
    {0.8, 0.1, 0.1},
    {0.1, 0.8, 0.1},
    {0.1, 0.1, 0.8},
    {0.5, 0.3, 0.2}
  };
  sls_t sls;
  double wmin, log_ratio, du, best_chi;
  double bestY[VISCO_SLS_MAX] = {0};
  int l, m, s;

  if(L < 1 || L > VISCO_SLS_MAX) return;
  if(!isfinite(fmin) || !isfinite(fmax) || !isfinite(Q0) ||
     fmin <= 0.0f || fmax < fmin || Q0 <= 0.0f) return;

  sls.L = L;
  sls.qinv = 1.0/(double)Q0;
  wmin = 2.0*pi*(double)fmin;
  log_ratio = log((double)fmax/(double)fmin);
  du = log_ratio/(VISCO_SLS_NODES-1);
  for(l=0; l<L; l++){
    double exponent = (L==1) ? 0.0 : (double)l/(L-1);
    wl[l] = wmin*exp(log_ratio*exponent);
  }
  for(m=0; m<VISCO_SLS_NODES; m++){
    double w = wmin*exp(m*du);
    /* A zero-width band is a single-frequency fit, with unit total weight. */
    sls.weight[m] = (fmin==fmax) ? 1.0/VISCO_SLS_NODES : du*w;
    if(fmin!=fmax && (m==0 || m==VISCO_SLS_NODES-1))
      sls.weight[m] *= 0.5;
    for(l=0; l<L; l++){
      double omega_l = wl[l];
      double denominator = omega_l*omega_l + w*w;
      sls.A[m][l] = omega_l*w/denominator;
      sls.B[m][l] = omega_l*omega_l/denominator;
    }
  }

  best_chi = sls_evaluate(&sls, bestY, 0, 0);
  for(s=0; s<5; s++){
    double Y[VISCO_SLS_MAX], chi;
    for(l=0; l<L; l++) Y[l] = starts[s][l];
    chi = sls_lm_solve(&sls, Y);
    if(chi < best_chi){
      best_chi = chi;
      for(l=0; l<L; l++) bestY[l] = Y[l];
    }
  }
  for(l=0; l<L; l++) yl[l] = bestY[l]*Q0;//convert Yl to yl=Yl*Q0
}
