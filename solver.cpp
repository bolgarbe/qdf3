#include "solver.h"
#include "external.h"
#include <cstdlib>
#include <time.h>

static const double tau=1e-6;

OneClassSolver::ReturnType OneClassSolver::solve(
        const SolverParams& params,
        Combination& co,
        const std::vector<OneClassSolver::TrainingType>& tr)
{
    Base::reset();
    clock_t start = clock();

    num_training = tr.size();
    num_kernels = co.size();

    alpha.resize(num_training);
    Q_alpha.resize(num_kernels*num_training,0.0);
    alpha_Q_alpha.resize(num_kernels,0.0);
    gradient.resize(num_training);
    H_i.resize(num_training);
    H_d.resize(num_training);
    check_bound.resize(num_training);
    auto& w = co.getWeights();
    auto& K = co.getKernels();
    double gap;

    // init buffers
    for(int i=0; i<num_kernels; i++)
        w[i]=1.0/num_kernels;
    int m = num_training*params.nu;
    for(int i=0; i<m; i++)
    {
        alpha[i]=1;
        check_bound[i]=1;
    }
    for(int i=m; i<num_training; i++)
    {
        alpha[i]=0;
        check_bound[i]=-1;
    }
    if(m<num_training)
    {
        alpha[m] = params.nu*num_training-m;
        check_bound[m]=0;
    }

    for(int i=0; i<num_training; i++)
    {
        if(check_bound[i]!=-1)
        {
            double alpha_i = alpha[i];
            for(int n=0; n<num_kernels; n++)
            {
                for(int j=0; j<num_training; j++)
                {
                    double tmp = alpha_i * K[n].k_func(tr[i].x,tr[j].x);
                    Q_alpha[n*num_training+j]+=tmp;
                    alpha_Q_alpha[n]+=tmp*alpha[j];
                }
            }
        }
    }

    for(int n=0; n<num_kernels; n++)
    {
        double aQa_n = alpha_Q_alpha[n] * -0.5;
        w[n]=-aQa_n/params.lambda;
        alpha_Q_alpha[n]=aQa_n;
    }

    int iters=0;
    while(true)
    {
        // Compute gradient & select 1st variable
        int i=-1;
        double b_up=-INFINITY;
        for(int t=0; t<num_training; t++)
        {
            double val=0.0;
            for(int n=0; n<num_kernels; n++)
            {
                val-=w[n]*Q_alpha[n*num_training+t];
            }
            gradient[t]=-val;

            bool is_up = check_bound[t] != 1;
            bool is_gt = (val>b_up) && is_up;
            b_up = is_gt ? val : b_up;
            i = is_gt ? t : i;
        }

        // Compute Hessian
        for(int t=0; t<num_training; t++)
        {
            double H_it=0.0,H_tt=0.0;
            double Q_it=0.0,Q_tt=0.0;
            for(int n=0; n<num_kernels; n++)
            {
                double QA_t = Q_alpha[n*num_training+t];
                H_it += Q_alpha[n*num_training+i]*QA_t;
                H_tt += QA_t*QA_t;
                Q_it += K[n].k_func(tr[i].x,tr[t].x)*w[n];
                Q_tt += K[n].k_func(tr[t].x,tr[t].x)*w[n];
            }
            H_i[t] = Q_it + H_it/params.lambda;
            H_d[t] = Q_tt + H_tt/params.lambda;
        }

        // Select 2nd variable
        int j=-1;
        double b_lo=INFINITY,conv=-INFINITY;
        double H_ii=H_i[i];
        for(int t=0; t<num_training; t++)
        {
            double grad = gradient[t];
            double b = b_up + grad;
            double H_it = H_i[t];
            double H_tt = H_d[t];
            double a = H_tt-2.0*H_it+H_ii;
            a = a > tau ? a : tau;
            double val = -b*b/a;

            bool is_lo = check_bound[t] != -1;
            bool is_gt = is_lo && grad>conv;
            bool is_lt = is_lo && b>0.0 && val<b_lo;

            conv = is_gt ? grad : conv;
            b_lo = is_lt ? val : b_lo;
            j = is_lt ? t : j;
        }

        gap=b_up+conv;
        if(gap<params.eps) break;

        // Solve subproblem
        double a_i=alpha[i],a_j=alpha[j];
        double grad_diff = gradient[i]-gradient[j];
        int grad_sign = (grad_diff>0.0) - (grad_diff<0.0);
        double dir_i = -grad_sign, dir_j=grad_sign;
        double a=0.0,b=0.0,c=0.0,d=-fabs(grad_diff);

        for(int n=0; n<num_kernels; n++)
        {
            double q=Q_alpha[n*num_training+i]*dir_i + Q_alpha[n*num_training+j]*dir_j;
            double r=K[n].k_func(tr[i].x,tr[i].x) + K[n].k_func(tr[j].x,tr[j].x) - K[n].k_func(tr[i].x,tr[j].x)*2.0;
            a+=r*r;
            b+=r*q;
            c+=q*q - alpha_Q_alpha[n]*r;
        }
        a/=params.lambda*8.0;
        b/=params.lambda*2.0;
        c/=params.lambda*2.0;

        double step=fmin(double(dir_i>0)-a_i*dir_i,
                         double(dir_j>0)-a_j*dir_j);
        bool is_zero = a==0.0 && b==0.0 && c==0.0;
        double max_root = is_zero ? step : solve_cubic(a*4.0,b*3.0,c*2.0,d);
        max_root = max_root>step ? step : max_root<0.0 ? 0.0 : max_root;

        double delta_i=max_root*dir_i, delta_j=max_root*dir_j;
        a_i+=delta_i; alpha[i]=a_i;
        a_j+=delta_j; alpha[j]=a_j;

        check_bound[i] = a_i>=1.0 ? 1 :
                         a_i<=0.0 ? -1 : 0;
        check_bound[j] = a_j>=1.0 ? 1 :
                         a_j<=0.0 ? -1 : 0;

        // Update buffers
        for(int n=0; n<num_kernels; n++)
        {
            double aQa_n = alpha_Q_alpha[n];

            aQa_n -= (0.5*K[n].k_func(tr[i].x,tr[i].x)*delta_i*delta_i +
                      0.5*K[n].k_func(tr[j].x,tr[j].x)*delta_j*delta_j +
                          K[n].k_func(tr[i].x,tr[j].x)*delta_i*delta_j);
            aQa_n -= (Q_alpha[n*num_training+i]*delta_i +
                      Q_alpha[n*num_training+j]*delta_j);
            alpha_Q_alpha[n] = aQa_n > 0.0 ? 0.0 : aQa_n;

            w[n]=-aQa_n/params.lambda;
        }

        for(int n=0; n<num_kernels; n++)
        {
            for(int t=0; t<num_training; t++)
            {
                Q_alpha[n*num_training+t] += (K[n].k_func(tr[i].x,tr[t].x)*delta_i +
                                              K[n].k_func(tr[j].x,tr[t].x)*delta_j);
            }
        }

        ++iters;
    }

    double obj=0.0,rank_coef=0.0;
    for(int n=0; n<num_kernels; n++)
    {
        double aQa_n = alpha_Q_alpha[n];
        obj += aQa_n*aQa_n;
        rank_coef += aQa_n*w[n];
    }
    obj/=2.0*params.lambda;
    rank_coef*=-2.0;
    rank_coef = 1.0/sqrt(rank_coef);

    double b_up = INFINITY;
    double b_lo = -INFINITY;
    double sum=0.0;
    int num_free=0;

    for(int t=0; t<num_training; t++)
    {
        double grad = gradient[t];
        signed char bound=check_bound[t];

        bool bnd = bound==1;
        bool fr  = bound==0;
        bool lt  = !bnd && grad<b_up;
        bool gt  = bnd && grad>b_lo;

        b_up = lt ? grad : b_up;
        b_lo = gt ? grad : b_lo;

        num_free += fr ? 1 : 0;
        sum += fr ? grad : 0;
    }
    double f_b = num_free>0 ? sum/num_free : (b_up+b_lo)/2.0;
    f_b*=rank_coef;

    double time = double(clock()-start)/CLOCKS_PER_SEC;

    ReturnType model(std::move(alpha),co,tr);
    model.time=time;
    model.iters=iters;
    model.obj=obj;
    model.b=0.0;
    model.margin=f_b;
    model.rank_coef=rank_coef;
    model.num_training=num_training;

    return model;
}

TwoClassSolver::ReturnType TwoClassSolver::solve(
        const SolverParams& params,
        Combination& co,
        const std::vector<TwoClassSolver::TrainingType>& tr)
{
    Base::reset();
    clock_t start = clock();

    num_training = tr.size();
    num_kernels = co.size();

    alpha.resize(num_training,0.0);
    Q_alpha.resize(num_kernels*num_training,0.0);
    alpha_Q_alpha.resize(num_kernels,0.0);
    gradient.resize(num_training);
    H_i.resize(num_training);
    H_d.resize(num_training);
    check_bound.resize(num_training);
    auto& w = co.getWeights();
    auto& K = co.getKernels();
    double gap;

    double alpha_sum=0.0;
    y.resize(num_training);

    // init buffers
    for(int i=0; i<num_kernels; i++)
        w[i]=0.0;

    for(int i=0; i<num_training; i++)
    {
        check_bound[i]=-1;
        y[i]=tr[i].y;
    }

    int iters=0;
    while(true)
    {
        // Compute gradient & select 1st variable
        int i=-1;
        double b_up=-INFINITY;
        for(int t=0; t<num_training; t++)
        {
            double val=1.0;
            for(int n=0; n<num_kernels; n++)
            {
                val-=w[n]*Q_alpha[n*num_training+t];
            }
            gradient[t]=-val;
            val*=y[t];

            bool is_up = check_bound[t] != y[t];
            bool is_gt = (val>b_up) && is_up;
            b_up = is_gt ? val : b_up;
            i = is_gt ? t : i;
        }

        // Compute Hessian
        for(int t=0; t<num_training; t++)
        {
            double H_it=0.0,H_tt=0.0;
            double Q_it=0.0,Q_tt=0.0;
            for(int n=0; n<num_kernels; n++)
            {
                double QA_t = Q_alpha[n*num_training+t];
                H_it += Q_alpha[n*num_training+i]*QA_t;
                H_tt += QA_t*QA_t;
                Q_it += K[n].k_func(tr[i].x,tr[t].x)*w[n]; // *y[i]*y[t] ?
                Q_tt += K[n].k_func(tr[t].x,tr[t].x)*w[n];
            }
            H_i[t] = Q_it + H_it/params.lambda;
            H_d[t] = Q_tt + H_tt/params.lambda;
        }

        // Select 2nd variable
        int j=-1;
        double b_lo=INFINITY,conv=-INFINITY;
        double H_ii=H_i[i];
        for(int t=0; t<num_training; t++)
        {
            double grad = gradient[t]*y[t];
            double b = b_up + grad;
            double H_it = H_i[t];
            double H_tt = H_d[t];
            double a = H_tt-2.0*H_it+H_ii;
            a = a > tau ? a : tau;
            double val = -b*b/a;

            bool is_lo = check_bound[t] != -y[t];
            bool is_gt = is_lo && grad>conv;
            bool is_lt = is_lo && b>0.0 && val<b_lo;

            conv = is_gt ? grad : conv;
            b_lo = is_lt ? val : b_lo;
            j = is_lt ? t : j;
        }

        gap=b_up+conv;
        if(gap<params.eps) break;

        // Solve subproblem
        double a_i=alpha[i],a_j=alpha[j];
        signed char y_ij=y[i]*y[j];
        double grad_diff = gradient[i]-gradient[j]*y_ij;
        int grad_sign = (grad_diff>0.0) - (grad_diff<0.0);
        double dir_i = -grad_sign, dir_j=grad_sign*y_ij;
        double a=0.0,b=0.0,c=0.0,d=-fabs(grad_diff);

        for(int n=0; n<num_kernels; n++)
        {
            double q=Q_alpha[n*num_training+i]*dir_i + Q_alpha[n*num_training+j]*dir_j;
            double r=K[n].k_func(tr[i].x,tr[i].x) + K[n].k_func(tr[j].x,tr[j].x) - K[n].k_func(tr[i].x,tr[j].x)*2.0*y_ij;
            a+=r*r;
            b+=r*q;
            c+=q*q - alpha_Q_alpha[n]*r;
        }
        a/=params.lambda*8.0;
        b/=params.lambda*2.0;
        c/=params.lambda*2.0;

        double step=fmin(params.c*double(dir_i>0)-a_i*dir_i,
                         params.c*double(dir_j>0)-a_j*dir_j);
        bool is_zero = a==0.0 && b==0.0 && c==0.0;
        double max_root = is_zero ? step : solve_cubic(a*4.0,b*3.0,c*2.0,d);
        max_root = max_root>step ? step : max_root<0.0 ? 0.0 : max_root;

        double delta_i=max_root*dir_i, delta_j=max_root*dir_j;
        a_i+=delta_i; alpha[i]=a_i;
        a_j+=delta_j; alpha[j]=a_j;
        alpha_sum+=delta_i+delta_j;

        check_bound[i] = a_i>=params.c ? 1 :
                         a_i<=0.0 ? -1 : 0;
        check_bound[j] = a_j>=params.c ? 1 :
                         a_j<=0.0 ? -1 : 0;

        // Update buffers
        for(int n=0; n<num_kernels; n++)
        {
            double aQa_n = alpha_Q_alpha[n];

            aQa_n -= (0.5*K[n].k_func(tr[i].x,tr[i].x)*delta_i*delta_i +
                      0.5*K[n].k_func(tr[j].x,tr[j].x)*delta_j*delta_j +
                          K[n].k_func(tr[i].x,tr[j].x)*delta_i*delta_j);
            aQa_n -= (Q_alpha[n*num_training+i]*delta_i +
                      Q_alpha[n*num_training+j]*delta_j);
            alpha_Q_alpha[n] = aQa_n > 0.0 ? 0.0 : aQa_n;

            w[n]=-aQa_n/params.lambda;
        }

        for(int n=0; n<num_kernels; n++)
        {
            for(int t=0; t<num_training; t++)
            {
                Q_alpha[n*num_training+t] += (K[n].k_func(tr[i].x,tr[t].x)*delta_i +
                                              K[n].k_func(tr[j].x,tr[t].x)*delta_j);
            }
        }

        ++iters;
    }

    double obj=0.0,rank_coef=0.0;
    for(int n=0; n<num_kernels; n++)
    {
        double aQa_n = alpha_Q_alpha[n];
        obj += aQa_n*aQa_n;
        rank_coef += aQa_n*w[n];
    }
    obj/=2.0*params.lambda;
    obj-=alpha_sum;
    rank_coef*=-2.0;
    rank_coef = 1.0/sqrt(rank_coef);

    double b_up = INFINITY;
    double b_lo = -INFINITY;
    double sum=0.0;
    int num_free=0;

    for(int t=0; t<num_training; t++)
    {
        signed char y_t=y[t];
        alpha[t]*=y_t;
        double grad=gradient[t]*y_t;
        signed char bound=check_bound[t];

        bool bnd = bound==y[t];
        bool fr  = bound==0;
        bool lt  = !bnd && grad<b_up;
        bool gt  = bnd && grad>b_lo;

        b_up = lt ? grad : b_up;
        b_lo = gt ? grad : b_lo;

        num_free += fr ? 1 : 0;
        sum += fr ? grad : 0;
    }
    double f_b = num_free>0 ? sum/num_free : (b_up+b_lo)/2.0;
    f_b*=rank_coef;

    double time = double(clock()-start)/CLOCKS_PER_SEC;

    ReturnType model(std::move(alpha),co,tr);
    model.time=time;
    model.iters=iters;
    model.obj=obj;
    model.b=-f_b;
    model.margin=f_b;
    model.rank_coef=rank_coef;
    model.num_training=num_training;

    return model;
}

SvrSolver::ReturnType SvrSolver::solve(
        const SolverParams& params,
        Combination& co,
        const std::vector<SvrSolver::TrainingType>& tr)
{
    Base::reset();
    clock_t start = clock();

    num_training = tr.size();
    num_kernels = co.size();
    int eff_training = num_training*2;

    alpha.resize(eff_training,0.0);
    Q_alpha.resize(num_kernels*eff_training,0.0);
    alpha_Q_alpha.resize(num_kernels,0.0);
    gradient.resize(eff_training);
    H_i.resize(eff_training);
    H_d.resize(eff_training);
    check_bound.resize(eff_training);
    auto& w = co.getWeights();
    auto& K = co.getKernels();
    double gap;

    double alpha_sum=0.0;
    y.resize(eff_training);

    // init buffers
    for(int i=0; i<num_kernels; i++)
        w[i]=0.0;

    for(int i=0; i<num_training; i++)
    {
        check_bound[i]=-1;
        check_bound[i+num_training]=-1;
        y[i]=params.r_eps-tr[i].y;
        y[i+num_training]=params.r_eps+tr[i].y;
    }

    int iters=0;
    while(true)
    {
        // Compute gradient & select 1st variable
        int i=-1;
        double b_up=-INFINITY;
        for(int t=0; t<eff_training; t++)
        {
            double val=-y[t];
            for(int n=0; n<num_kernels; n++)
            {
                val-=w[n]*Q_alpha[n*eff_training+t];
            }
            gradient[t]=-val;
            signed char sy_t = (t<num_training ? 1 : -1);
            val *= sy_t;

            bool is_up = check_bound[t] != sy_t;
            bool is_gt = (val>b_up) && is_up;
            b_up = is_gt ? val : b_up;
            i = is_gt ? t : i;
        }

        // Compute Hessian
        signed char i_pos = i<num_training ? 1 : -1;
        int ii = i<num_training ? i : i-num_training;

        for(int t=0; t<num_training; t++)
        {
            double H_it1=0.0,H_it2=0.0,H_tt1=0.0,H_tt2=0.0;
            double Q_it=0.0,Q_tt=0.0;
            for(int n=0; n<num_kernels; n++)
            {
                int idx=n*eff_training;

                double QA_t1 = Q_alpha[idx+t];
                double QA_t2 = Q_alpha[idx+t+num_training];
                H_it1 += Q_alpha[idx+i]*QA_t1;
                H_it2 += Q_alpha[idx+i+num_training]*QA_t2;
                H_tt1 += QA_t1*QA_t1;
                H_tt2 += QA_t2*QA_t2;
                Q_it += K[n].k_func(tr[ii].x,tr[t].x)*w[n];
                Q_tt += K[n].k_func(tr[t].x,tr[t].x)*w[n];
            }
            H_i[t] = Q_it*i_pos + H_it1/params.lambda;
            H_i[t+num_training] = -Q_it*i_pos + H_it2/params.lambda;
            H_d[t] = Q_tt + H_tt1/params.lambda;
            H_d[t+num_training] = Q_tt + H_tt2/params.lambda;
        }

        // Select 2nd variable
        int j=-1;
        double b_lo=INFINITY,conv=-INFINITY;
        double H_ii=H_d[i];
        for(int t=0; t<eff_training; t++)
        {
            signed char sy_t = (t<num_training ? 1 : -1);
            double grad = gradient[t]*sy_t;
            double b = b_up + grad;
            double H_it = H_i[t];
            double H_tt = H_d[t];
            double a = H_tt-2.0*H_it*i_pos*sy_t+H_ii;
            a = a > tau ? a : tau;
            double val = -b*b/a;

            bool is_lo = check_bound[t] != -sy_t;
            bool is_gt = is_lo && grad>conv;
            bool is_lt = is_lo && b>0.0 && val<b_lo;

            conv = is_gt ? grad : conv;
            b_lo = is_lt ? val : b_lo;
            j = is_lt ? t : j;
        }

        gap=b_up+conv;
        if(gap<params.eps) break;

        // Solve subproblem
        signed char j_pos = j<num_training ? 1:-1;
        int jj = j<num_training ? j : j-num_training;

        double a_i=alpha[i],a_j=alpha[j];
        signed char y_ij=i_pos*j_pos;
        double grad_diff = gradient[i]-gradient[j]*y_ij;
        int grad_sign = (grad_diff>0.0) - (grad_diff<0.0);
        double dir_i = -grad_sign, dir_j=grad_sign*y_ij;
        double a=0.0,b=0.0,c=0.0,d=-fabs(grad_diff);

        for(int n=0; n<num_kernels; n++)
        {
            double q=Q_alpha[n*eff_training+i]*dir_i + Q_alpha[n*eff_training+j]*dir_j;
            double r=K[n].k_func(tr[ii].x,tr[ii].x) + K[n].k_func(tr[jj].x,tr[jj].x) - K[n].k_func(tr[ii].x,tr[jj].x)*2.0;
            a+=r*r;
            b+=r*q;
            c+=q*q - alpha_Q_alpha[n]*r;
        }
        a/=params.lambda*8.0;
        b/=params.lambda*2.0;
        c/=params.lambda*2.0;

        double step=fmin(params.c*double(dir_i>0)-a_i*dir_i,
                         params.c*double(dir_j>0)-a_j*dir_j);
        bool is_zero = a==0.0 && b==0.0 && c==0.0;
        double max_root = is_zero ? step : solve_cubic(a*4.0,b*3.0,c*2.0,d);
        max_root = max_root>step ? step : max_root<0.0 ? 0.0 : max_root;

        double delta_i=max_root*dir_i, delta_j=max_root*dir_j;
        a_i+=delta_i; alpha[i]=a_i;
        a_j+=delta_j; alpha[j]=a_j;
        alpha_sum+=delta_i+delta_j;

        check_bound[i] = a_i>=params.c ? 1 :
                         a_i<=0.0 ? -1 : 0;
        check_bound[j] = a_j>=params.c ? 1 :
                         a_j<=0.0 ? -1 : 0;

        // Update buffers
        for(int n=0; n<num_kernels; n++)
        {
            double aQa_n = alpha_Q_alpha[n];

            aQa_n -= (0.5*K[n].k_func(tr[ii].x,tr[ii].x)*delta_i*delta_i +
                      0.5*K[n].k_func(tr[jj].x,tr[jj].x)*delta_j*delta_j +
                          K[n].k_func(tr[ii].x,tr[jj].x)*delta_i*delta_j*y_ij);
            aQa_n -= (Q_alpha[n*eff_training+i]*delta_i +
                      Q_alpha[n*eff_training+j]*delta_j);
            alpha_Q_alpha[n] = aQa_n > 0.0 ? 0.0 : aQa_n;

            w[n]=-aQa_n/params.lambda;
        }

        for(int n=0; n<num_kernels; n++)
        {
            for(int t=0; t<num_training; t++)
            {
                Q_alpha[n*eff_training+t] += (K[n].k_func(tr[ii].x,tr[t].x)*delta_i*i_pos +
                                              K[n].k_func(tr[jj].x,tr[t].x)*delta_j*j_pos);

                Q_alpha[n*eff_training+num_training+t] -= (K[n].k_func(tr[ii].x,tr[t].x)*delta_i*i_pos +
                                                           K[n].k_func(tr[jj].x,tr[t].x)*delta_j*j_pos);
            }
        }

        ++iters;
    }

    double obj=0.0,rank_coef=0.0;
    for(int n=0; n<num_kernels; n++)
    {
        double aQa_n = alpha_Q_alpha[n];
        obj += aQa_n*aQa_n;
        rank_coef += aQa_n*w[n];
    }
    obj/=2.0*params.lambda;
    obj-=alpha_sum;
    rank_coef*=-2.0;
    rank_coef = 1.0/sqrt(rank_coef);

    double b_up = INFINITY;
    double b_lo = -INFINITY;
    double sum=0.0;
    int num_free=0;

    for(int t=0; t<eff_training; t++)
    {
        signed char y_t = t<num_training ? 1 : -1;
        double grad=gradient[t]*y_t;
        signed char bound=check_bound[t];

        bool bnd = bound==y[t];
        bool fr  = bound==0;
        bool lt  = !bnd && grad<b_up;
        bool gt  = bnd && grad>b_lo;

        b_up = lt ? grad : b_up;
        b_lo = gt ? grad : b_lo;

        num_free += fr ? 1 : 0;
        sum += fr ? grad : 0;
    }
    double f_b = num_free>0 ? sum/num_free : (b_up+b_lo)/2.0;

    double time = double(clock()-start)/CLOCKS_PER_SEC;

    std::vector<double> alpha_eff(num_training);
    for(int t=0; t<num_training; t++)
        alpha_eff[t]=alpha[t]-alpha[t+num_training];

    MklModel model(std::move(alpha_eff),co,tr);
    model.time=time;
    model.iters=iters;
    model.obj=obj;
    model.b=-f_b;
    model.rank_coef=1.0;
    model.num_training=num_training;

    return model;
}

DmlSolver::ReturnType DmlSolver::solve(
        const SolverParams& params,
        Combination& co,
        const std::vector<DmlSolver::TrainingType>& tr)
{
    Base::reset();
    clock_t start=clock();
    num_training = tr.size();
    num_kernels = co.size();

    alpha.resize(num_training,0.0);
    Q_alpha.resize(num_kernels*num_training,0.0);
    alpha_Q_alpha.resize(num_kernels,0.0);
    dist.resize(num_training);
    Q_i.resize(num_training*num_kernels);
    y.resize(num_training);
    auto& w = co.getWeights();
    auto& K = co.getKernels();

    for(int n=0; n<num_kernels; n++)
    {
        w[n]=0.0;
        K[n].computeKernelized();
    }

    for(int t=0; t<num_training; t++)
    {
        dist[t]=0.0;
        for(int n=0; n<num_kernels; n++)
            dist[t]+=K[n].dist(tr[t].x,tr[t].z);
        y[t]=tr[t].y;
    }

    int iters=0;
    while(true)
    {
        double d_max=-INFINITY;
        for(int i=0; i<num_training; i++)
        {
            double a=0.0,b=0.0,c=0.0;
            double d=params.rho-(params.r-dist[i])*y[i];

            for(int n=0; n<num_kernels; n++)
            {
                for(int t=0; t<num_training; t++)
                    Q_i[n*num_training+t]=K[n].Q_func(tr[i],tr[t]);
                double r=Q_i[n*num_training+i];
                double q=Q_alpha[n*num_training+i]*2.0;

                a+=r*r;
                b+=q*r;
                c+=q*q+2.0*r*alpha_Q_alpha[n];
                d-=w[n]*Q_alpha[n*num_training+i];
            }
            a/=-params.lambda*2.0;
            b=-b*3.0/(4.0*params.lambda);
            c/=-params.lambda*4.0;

            double max_root=solve_cubic(a,b,c,d);
            double a_i=alpha[i];
            /*double delta = a_i+max_root > params.c ? params.c-a_i :
                           a_i+max_root < 0.0 ? -a_i : max_root;*/
            double delta = y[i]>0 ? a_i+max_root > params.c ? params.c-a_i :
                                    a_i+max_root < 0.0 ? -a_i : max_root
                                  :
                                    a_i+max_root > params.cn ? params.cn-a_i :
                                    a_i+max_root < 0.0 ? -a_i : max_root;
            alpha[i]=a_i+delta;

            double d_abs=fabs(delta);
            d_max = d_abs>d_max ? d_abs : d_max;

            for(int n=0; n<num_kernels; n++)
            {
                double aQa_n = alpha_Q_alpha[n];
                aQa_n+=Q_i[n*num_training+i]*delta*delta;
                aQa_n+=2.0*Q_alpha[n*num_training+i]*delta;

                w[n]=aQa_n/(params.lambda*2.0);
                alpha_Q_alpha[n]=aQa_n;

                for(int t=0; t<num_training; t++)
                    Q_alpha[n*num_training+t]+=Q_i[n*num_training+t]*delta*y[i]*y[t];
            }
        }

        ++iters;
        if(d_max<params.eps) break;
    }

    double obj=0.0;
    for(int n=0; n<num_kernels; n++)
    {
        double aQa_n = alpha_Q_alpha[n];
        obj+=aQa_n*aQa_n;
    }
    obj*=-1.0/(8.0*params.lambda);

    for(int t=0; t<num_training; t++)
    {
        double a_t=alpha[t]*y[t];
        obj+=a_t*(params.r-dist[t]);
        alpha[t]=a_t;
    }
    obj+=params.rho*num_training;

    double time = double(clock()-start)/CLOCKS_PER_SEC;

    ReturnType model(std::move(alpha),co,tr);
    model.time=time;
    model.iters=iters;
    model.obj=obj;

    return model;
}

DmlRSolver::ReturnType DmlRSolver::solve(
        const SolverParams& params,
        Combination& co,
        const std::vector<DmlRSolver::TrainingType>& tr)
{
    Base::reset();
    clock_t start=clock();
    num_training = tr.size();
    num_kernels = co.size();
    int eff_training=num_training*2;

    alpha.resize(eff_training,0.0);
    Q_alpha.resize(num_kernels*num_training,0.0);
    alpha_Q_alpha.resize(num_kernels,0.0);
    dist.resize(num_training);
    Q_i.resize(num_training*num_kernels);
    y.resize(num_training);
    auto& w = co.getWeights();
    auto& K = co.getKernels();

    for(int n=0; n<num_kernels; n++)
    {
        w[n]=0.0;
        K[n].computeKernelized();
    }

    for(int t=0; t<num_training; t++)
    {
        dist[t]=0.0;
        for(int n=0; n<num_kernels; n++)
            dist[t]+=K[n].dist(tr[t].x,tr[t].z);
        y[t]=tr[t].y;
    }

    int iters=0;
    while(true)
    {
        double d_max=-INFINITY;
        for(int i=0; i<num_training; i++)
        {
            double a=0.0,b=0.0,c=0.0;
            double d1=dist[i]-y[i]-params.r_eps;
            double d2=y[i]-dist[i]-params.r_eps;

            for(int n=0; n<num_kernels; n++)
            {
                for(int t=0; t<num_training; t++)
                    Q_i[n*num_training+t]=K[n].Q_func(tr[i],tr[t]);
                double r=Q_i[n*num_training+i];
                double q=Q_alpha[n*num_training+i]*2.0;

                a+=r*r;
                b+=q*r;
                c+=q*q+2.0*r*alpha_Q_alpha[n];
                double d=w[n]*Q_alpha[n*num_training+i];
                d1-=d;
                d2+=d;
            }
            a/=-params.lambda*2.0;
            b=-b*3.0/(4.0*params.lambda);
            c/=-params.lambda*4.0;

            double max_root1=solve_cubic(a,b,c,d1);
            double max_root2=solve_cubic(a,-b,c,d2);
            double a_i1=alpha[i],a_i2=alpha[i+num_training];
            double delta1 = a_i1+max_root1 > params.c ? params.c-a_i1 :
                            a_i1+max_root1 < 0.0 ? -a_i1 : max_root1;
            double delta2 = a_i2+max_root2 > params.c ? params.c-a_i2 :
                            a_i2+max_root2 < 0.0 ? -a_i2 : max_root2;
            alpha[i]=a_i1+delta1;
            alpha[i+num_training]=a_i2+delta2;

            double delta=delta1>delta2 ? delta1:delta2;
            double d_abs=fabs(delta);
            d_max = d_abs>d_max ? d_abs : d_max;

            for(int n=0; n<num_kernels; n++)
            {
                double aQa_n = alpha_Q_alpha[n];
                aQa_n+=Q_i[n*num_training+i]*(delta1*delta1 + delta2*delta2);
                aQa_n+=2.0*Q_alpha[n*num_training+i]*(delta1-delta2);

                w[n]=aQa_n/(params.lambda*2.0);
                alpha_Q_alpha[n]=aQa_n;

                for(int t=0; t<num_training; t++)
                    Q_alpha[n*num_training+t]+=Q_i[n*num_training+t]*(delta1-delta2);
            }
        }

        ++iters;
        if(d_max<params.eps) break;
    }

    std::vector<double> alpha_eff(num_training);
    for(int t=0; t<num_training; t++)
        alpha_eff[t]=alpha[t]-alpha[t+num_training];

    double time = double(clock()-start)/CLOCKS_PER_SEC;

    ReturnType model(std::move(alpha_eff),co,tr);
    model.time=time;
    model.iters=iters;

    return model;
}
