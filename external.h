#ifndef EXTERNAL_H
#define EXTERNAL_H

#include <math.h>

double SQRT2  =  1.41421356237309504880;       /* sqrt(2) */
double SQRTH  =  7.07106781186547524401E-1;    /* sqrt(2)/2 */
double LOG2E  =  1.4426950408889634073599;     /* 1/log(2) */
double SQ2OPI =  7.9788456080286535587989E-1;  /* sqrt( 2/pi ) */

double ndtr(double a)
{
double x, y, z;

x = a * SQRTH;
z = fabs(x);

/* if( z < SQRTH ) */
if( z < 1.0 )
    y = 0.5 + 0.5 * erf(x);

else
    {
#ifdef USE_EXPXSQ
    /* See below for erfce. */
    y = 0.5 * erfce(z);
    /* Multiply by exp(-x^2 / 2)  */
    z = expx2(a, -1);
    y = y * sqrt(z);
#else
    y = 0.5 * erfc(z);
#endif
    if( x > 0 )
        y = 1.0 - y;
    }

return(y);
}

double solve_cubic(double a, double b, double c, double d)
{
    double roots[3];
    double max_root=0.0;

    if(a==0.0 && b==0.0 && c==0.0)
    {
      max_root = INFINITY;
    }
    else
    {
      int num_roots;
      if(a==0.0)
      {
        if(b==0.0)
        {
          if(c==0.0) num_roots=0;
          else
          {
            roots[0]=-d/c;
            num_roots=1;
          }
        }
        else
        {
          const double det=c*c-b*d*4.0;
          int n=0;
          if(det>=0.0)
          {
            roots[n]=sqrt(det);
            if(det>0.0)
            {
              roots[n+1]=-roots[n];
              roots[n]-=c;
              roots[n]/=b*2.0;
              ++n;
            }
            roots[n]-=c;
            roots[n]/=b*2.0;
            ++n;
          }
          num_roots=n;
        }
      }
      else
      {
        int n=0;
        const double a3 = 1.0/(3.0*a);
        const double ba3 = b*a3;
        const double q=c*a3 - ba3*ba3;
        const double r=(ba3*c-d)/(a*2.0) - ba3*ba3*ba3;
        const double det = q*q*q + r*r;

        if(det>0.0)
        {
          const double s = sqrt(det);
          roots[n]=cbrt(r+s)+cbrt(r-s)-ba3;
          ++n;
        }
        else if(det==0.0)
        {
          const double s = cbrt(r);
          roots[n]=s*2.0-ba3;
          ++n;
          if(s>0.0)
          {
            roots[n]=-s-ba3;
            ++n;
          }
        }
        else
        {
          const double rho = cbrt(sqrt(r*r-det));
          const double theta = atan2(sqrt(-det),r)/3.0;
          const double spt = rho*cos(theta);
          const double smt = rho*sin(theta)*sqrt(3.0);

          roots[n]=spt*2.0-ba3;
          ++n;
          roots[n]=-spt-ba3;
          roots[n+1]=roots[n];
          roots[n]+=smt;
          ++n;
          roots[n]-=smt;
          ++n;
        }
        num_roots=n;
      }
      for(int k=0; k<num_roots; k++)
      {
        double r_k = roots[k];
        max_root = r_k>max_root ? r_k : max_root;
      }
    }
    return max_root;
}

#endif // EXTERNAL_H
