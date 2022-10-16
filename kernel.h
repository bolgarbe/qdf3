#ifndef KERNEL_H
#define KERNEL_H

#include "exceptions.h"
#include "file_handler.h"

#include <cmath>
#include <memory>

using qdf_key_t = std::string;
using qdf_map_t = std::map<qdf_key_t,int>;

struct DmlTrainingSample
{
  template<typename T,typename R>
  DmlTrainingSample(T&& xx, R&& zz, double yy) noexcept: x(xx), z(zz), y(yy) {}

  qdf_key_t x, z;
  double y;
};

struct MklTrainingSample
{
  template<typename T>
  MklTrainingSample(T&& xx, double yy) noexcept: x(xx), y(yy) {}

  qdf_key_t x;
  double y;
};

struct StoreAll
{
  // TODO: tag dispatch to handle precomputed kernel handlers
  template<typename T>
  inline void assignHandler(T&& file)
  {
    handler = std::forward<T>(file);
    const auto& d = handler->getData();
    dots = d * d.transpose();
  }

  template<typename T>
  inline void assignSet(T&& training)
  {

  }

  template<typename T,typename R>
  inline double dot(T&& x, R&& y) const
  {
    int a = handler->getPos(std::forward<T>(x));
    int b = handler->getPos(std::forward<R>(y));

    // TODO: use kernelized if available
    return dots.coeff(a, b);
  }

  template<typename T>
  inline double sdot(T&& x) const
  {
    return dot(x,x);
  }

protected:
  Eigen::SparseMatrix<double> dots;
  std::unique_ptr<KernelFile> handler;
};

struct StoreDots
{
  template<typename T>
  inline void assignHandler(T&& file)
  {
    handler = std::forward<T>(file);
  }

  template<typename T>
  inline void assignSet(T&& training)
  {
    //TODO: fix size if sample not found
    positions.clear();

    auto names = handler->getNames();
    auto data = handler->getData().toDense();
    int sz = names.size();
    dots.resize(sz,training.size());
    diag.resize(sz);

    for(int i=0; i<sz; i++)
    {
        diag(i) = data.row(i).squaredNorm();
    }

    int c=0;
    for(const auto& i:training)
    {
      if(handler->contains(i))
      {
        dots.col(c)=data*(data.row(handler->getPos(i)).transpose());
        positions.insert(std::make_pair(i,c++));
      }
    }
  }

  template<typename T,typename R>
  inline double dot(T&& x, R&& y) const
  {
    int a = positions.at(std::forward<T>(x));
    int b = handler->getPos(std::forward<R>(y));

    // TODO: use kernelized if available
    return dots.coeff(b, a);
  }

  template<typename T>
  inline double sdot(T&& x) const
  {
    int a=handler->getPos(std::forward<T>(x));
    return diag(a);
  }

protected:
  Eigen::MatrixXd dots;
  Eigen::VectorXd diag;
  std::unique_ptr<KernelFile> handler;
  qdf_map_t positions;
};

struct DontStoreDots
{
  template<typename T>
  inline void assignHandler(T&& file)
  {
    data = file.getData().toDense();
  }

  inline double dot(int a, int b) const
  {
    return data.row(a).dot(data.row(b));
  }

protected:
  Eigen::MatrixXd data;
};

template <typename StorePolicy>
class Kernel: public StorePolicy
{
  using StorePolicy::dot;
  using StorePolicy::sdot;
  using StorePolicy::handler;
  using StorePolicy::assignHandler;

public:
  enum class kernel_type
  {
    COSINE, TANIMOTO, RBF, LINEAR, QUADRATIC, POLY
  };

  Kernel();
  Kernel(std::unique_ptr<KernelFile>&& file, kernel_type t = kernel_type::COSINE, double a = 0, double g = 1, int p = 2);
  Kernel(Kernel&& k) = default;
  ~Kernel() = default;

  inline void computeKernelized();

  template<typename T,typename R>
  inline double k_func(T&& a, R&& b) const;

  template<typename T,typename R>
  inline double dist(T&& x, R&& z) const;

  template<typename T,typename R>
  inline double Q_func(T&& a, R&& b) const;

  template<typename T>
  inline bool contains(T&& t) const;

  inline void setParam(double p);

  inline auto getNames() const;

private:
  kernel_type type;

  Eigen::MatrixXd kernelized_vectors;
  Eigen::MatrixXd kernelized_dots;
  Eigen::MatrixXd dots;

  double avg;
  double gamma;
  int deg;
  int num_training;
  bool kernelized;
};

template<typename StorePolicy>
Kernel<StorePolicy>::Kernel(): type(kernel_type::LINEAR),
  avg(0), gamma(0.1), num_training(0), kernelized(false) {}

template<typename StorePolicy>
Kernel<StorePolicy>::Kernel(std::unique_ptr<KernelFile>&& file, kernel_type t, double a, double g, int p):
  type(t), avg(a), gamma(g), deg(p), num_training(0), kernelized(false)
{
  assignHandler(std::move(file));
}

template<typename StorePolicy>
inline void Kernel<StorePolicy>::computeKernelized()
{
  if(kernelized) return;
  int size = handler->getSamplesNum(); // do some PCA here
  kernelized_vectors.resize(size, size);

  const auto& pos=handler->getPositions();

  for(const auto& s : pos)
  {
    int ctr = 0;
    std::string id = s.first;
    for(const auto& i : pos)
    {
      kernelized_vectors(s.second, ctr) = k_func(id, i.first);
      ++ctr;
    }
  }

  kernelized_dots = kernelized_vectors * kernelized_vectors.transpose();
  kernelized = true;
}

template<typename StorePolicy>
template<typename T,typename R>
inline double Kernel<StorePolicy>::k_func(T&& a, R&& b) const
{
  if(!(contains(a) && contains(b))) return avg;
  switch(type)
  {
  case kernel_type::LINEAR:
    return dot(a,b);
  case kernel_type::TANIMOTO:
  {
    double dot_ab = dot(a,b);
    return dot_ab / (sdot(a) + sdot(b) - dot_ab);
  }
  case kernel_type::RBF:
  {
    return exp(-gamma * (sdot(a) - dot(a,b) * 2.0 + sdot(b)));
  }
  case kernel_type::COSINE:
  {
    return dot(a,b) / (sqrt(sdot(a)) * sqrt(sdot(b)));
  }
  case kernel_type::QUADRATIC:
  {
    return pow(dot(a,b) / (sqrt(sdot(a)) * sqrt(sdot(b))), 2);
  }
  case kernel_type::POLY:
  {
    return pow(dot(a,b) / (sqrt(sdot(a)) * sqrt(sdot(b))) + 1, deg);
  }
  default:
    return dot(a,b);
  }
}

template<typename StorePolicy>
template<typename T,typename R>
inline double Kernel<StorePolicy>::dist(T&& x, R&& z) const
{
  int pos_x = handler->getPos(std::forward<T>(x));
  int pos_z = handler->getPos(std::forward<R>(z));

  return kernelized_dots(pos_x, pos_x) - 2.0 * kernelized_dots(pos_x, pos_z) + kernelized_dots(pos_z, pos_z);
}

template<typename StorePolicy>
template<typename T,typename R>
inline double Kernel<StorePolicy>::Q_func(T&& a, R&& b) const
{
  int ax = handler->getPos(a.x);
  int az = handler->getPos(a.z);
  int bx = handler->getPos(b.x);
  int bz = handler->getPos(b.z);

  double diff_prod = kernelized_dots(ax, bx) - kernelized_dots(ax, bz) - kernelized_dots(az, bx) + kernelized_dots(az, bz);
  return diff_prod * diff_prod;
}

template<typename StorePolicy>
template<typename T>
inline bool Kernel<StorePolicy>::contains(T&& t) const
{
  return handler->contains(std::forward<T>(t));
}

template<typename StorePolicy>
inline auto Kernel<StorePolicy>::getNames() const
{
  return handler->getNames();
}

template<typename StorePolicy>
inline void Kernel<StorePolicy>::setParam(double p)
{
  switch(type)
  {
  case kernel_type::RBF:
    gamma=p;
    break;
  case kernel_type::POLY:
    deg=p;
    break;
  }
}

#endif //KERNEL_H
