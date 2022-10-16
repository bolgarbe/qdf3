#ifndef MKLSOLVER_H
#define MKLSOLVER_H

#include "combination.h"

// TODO: factor this out
struct SolverParams
{
  double c,cn,nu,eps,lambda,r_eps,r,rho;
  int max_iters;
};

// TODO: this one too

class OneClassSolver;
class TwoClassSolver;
class DmlSolver;
class SvrSolver;
class DmlRSolver;
class MklModel;
class DmlModel;

template<typename T>
struct SolverTraits
{
    using TrainingType = MklTrainingSample;
    using ReturnType = MklModel;
};

template<>
struct SolverTraits<DmlSolver>
{
    using TrainingType = DmlTrainingSample;
    using ReturnType = DmlModel;
};

template<>
struct SolverTraits<DmlRSolver>
{
    using TrainingType = DmlTrainingSample;
    using ReturnType = DmlModel;
};

template<typename T>
class Model
{
public:
    template<typename S>
    Model(S&& a, const Combination& c) noexcept:alpha(std::forward<S>(a)),co(c),num_kernels(c.size()) {}
    virtual ~Model() = default;

    // INTERFACE
    template<typename S>
    inline double score(S&& s) const
    {
        return static_cast<const T*>(this)->score(std::forward<S>(s));
    }

    int getIters() const {return iters;}

    const auto& getWeights() const {return co.getWeights();}
    const auto& getAlphas() const {return alpha;}

protected:
    std::vector<double> alpha;
    const Combination& co;
    int iters,num_kernels;
    double time,obj,b,rank_coef,margin;
};

class MklModel:public Model<MklModel>
{
    friend class OneClassSolver;
    friend class TwoClassSolver;
    friend class SvrSolver;

public:
    template<typename S>
    MklModel(S&& a, const Combination& c, const std::vector<MklTrainingSample>& t) noexcept:
        Model(std::forward<S>(a),c),tr(t),num_training(t.size()){}

    template<typename S>
    inline double score(S&& s) const
    {
        double score=b;
        for(int t=0; t<num_training; t++)
        {
            const std::string& tr_smp = tr[t].x;
            double k_val=0.0;
            for(int n=0; n<num_kernels; n++)
            {
                k_val+=co.getKernel(n).k_func(tr_smp,s)*co.getWeight(n);
            }
            score+=k_val*alpha[t];
        }
        score*=rank_coef;
        return score;
    }

    template<typename S>
    inline Eigen::VectorXd scores(S&& all_samples)
    {
      int num_all = all_samples.size();
      Eigen::VectorXd sc = Eigen::VectorXd::Ones(num_all)*b;
      Eigen::Map<Eigen::VectorXd> al(&alpha[0],num_training);

      for(int n=0; n<num_kernels; n++)
      {
        Eigen::MatrixXd slice(num_all, num_training);

        int i=0;
        for(const auto& si:all_samples)
        {
          for(int j=0; j<num_training; j++)
          {
            slice(i,j)=co.getKernel(n).k_func(tr[j].x,si);
          }
          ++i;
        }

        sc+=(slice*al)*co.getWeight(n);
      }

      sc*=rank_coef;

      return sc;
    }

private:
    const std::vector<MklTrainingSample>& tr;
    int num_training;
};

class DmlModel:public Model<DmlModel>
{
    friend class DmlSolver;
    friend class DmlRSolver;
public:
    template<typename S>
    DmlModel(S&& a, const Combination& c, const std::vector<DmlTrainingSample>& t):
        Model(std::forward<S>(a),c),tr(t),num_training(t.size())
    {}

    template<typename S>
    inline double score(S&& s) const
    {
        double score=0.0;
        int c=0;
        for(const auto& t:query)
        {
            ++c;
            score+=dist(s,t);
        }
        return score/c;
    }

    template<typename S,typename R>
    inline double dist(S&& s1, R&& s2) const
    {
        if(!(co.allContains(s1) && co.allContains(s2))) return INFINITY;
        DmlTrainingSample test(s1,s2,0);
        double dist=0.0;
        for(int n=0; n<num_kernels; n++)
        {
            dist+=co.getKernel(n).dist(s1,s2);

            for(int t=0; t<num_training; t++)
            {
                dist-=co.getKernel(n).Q_func(test,tr[t])*alpha[t]*co.getWeight(n);
            }
        }
        return dist;
    }

    template<typename S>
    inline void setQuery(S&& q)
    {
        query=std::forward<S>(q);
    }

private:
    const std::vector<DmlTrainingSample>& tr;
    std::vector<std::string> query;
    int num_training;
};

template<typename SolverImpl>
class Solver
{
public:
    Solver() = default;

    inline auto solve(
            const SolverParams& params,
            Combination& co,
            const std::vector<typename SolverTraits<SolverImpl>::TrainingType>& tr)
    {
        //reset();
        return static_cast<SolverImpl*>(this)->solve(params,co,tr);
    }

    inline void reset()
    {
      alpha.clear();
      Q_alpha.clear();
      alpha_Q_alpha.clear();
      num_training=num_kernels=0;
      static_cast<SolverImpl*>(this)->reset();
    }

protected:
    std::vector<double> alpha,Q_alpha,alpha_Q_alpha;
    int num_training,num_kernels;
    double gap;
};

class OneClassSolver:public Solver<OneClassSolver>
{
public:
    using Base = Solver<OneClassSolver>;
    using ReturnType = SolverTraits<OneClassSolver>::ReturnType;
    using TrainingType = SolverTraits<OneClassSolver>::TrainingType;

    ReturnType solve(const SolverParams& params, Combination& co, const std::vector<TrainingType>& tr);

    inline void reset()
    {
      gradient.clear();
      H_i.clear();
      H_d.clear();
      check_bound.clear();
    }

private:
    std::vector<double> gradient,H_i,H_d;
    std::vector<signed char> check_bound;
};

class TwoClassSolver:public Solver<TwoClassSolver>
{
public:
    using Base = Solver<TwoClassSolver>;
    using ReturnType = SolverTraits<TwoClassSolver>::ReturnType;
    using TrainingType = SolverTraits<TwoClassSolver>::TrainingType;

    ReturnType solve(const SolverParams& params, Combination& co, const std::vector<TrainingType>& tr);

    inline void reset()
    {
      gradient.clear();
      H_i.clear();
      H_d.clear();
      check_bound.clear();
      y.clear();
    }

private:
    std::vector<double> gradient,H_i,H_d;
    std::vector<signed char> check_bound,y;
};

class SvrSolver:public Solver<SvrSolver>
{
public:
    using Base = Solver<SvrSolver>;
    using ReturnType = SolverTraits<SvrSolver>::ReturnType;
    using TrainingType = SolverTraits<SvrSolver>::TrainingType;

    ReturnType solve(const SolverParams& params, Combination& co, const std::vector<TrainingType>& tr);

    inline void reset()
    {
      gradient.clear();
      H_i.clear();
      H_d.clear();
      y.clear();
      check_bound.clear();
    }

private:
    std::vector<double> gradient,H_i,H_d,y;
    std::vector<signed char> check_bound;
};

class DmlSolver:public Solver<DmlSolver>
{
public:
    using Base = Solver<DmlSolver>;
    using ReturnType = SolverTraits<DmlSolver>::ReturnType;
    using TrainingType = SolverTraits<DmlSolver>::TrainingType;

    ReturnType solve(const SolverParams& params, Combination& co, const std::vector<TrainingType>& tr);

    inline void reset()
    {
      dist.clear();
      Q_i.clear();
      y.clear();
    }

private:
    std::vector<double> dist,Q_i;
    std::vector<signed char> y;
};

class DmlRSolver:public Solver<DmlRSolver>
{
public:
    using Base = Solver<DmlRSolver>;
    using ReturnType = SolverTraits<DmlRSolver>::ReturnType;
    using TrainingType = SolverTraits<DmlRSolver>::TrainingType;

    ReturnType solve(const SolverParams& params, Combination& co, const std::vector<TrainingType>& tr);

    inline void reset()
    {
      dist.clear();
      Q_i.clear();
      y.clear();
    }

private:
    std::vector<double> dist,Q_i,y;
};


#endif // MKLSOLVER_H
