#ifndef COMBINATION_H
#define COMBINATION_H

#include "kernel.h"

// Kernel combination
class Combination
{
public:
  inline Combination();
  ~Combination() = default;

  inline int size() const;

  inline void addKernel(std::unique_ptr<KernelFile>&& file, Kernel<StoreDots>::kernel_type type, double gamma = 1, double avg = 0);

  inline auto& getWeights();

  inline const auto& getWeights() const;

  inline auto& getKernels();

  inline const auto& getKernels() const;

  inline auto& getKernel(int n);

  inline const auto& getKernel(int n) const;

  inline double getWeight(int n) const;

  template<typename T, typename R>
  inline double k_func(T&& a, R&& b) const;

  template<typename T, typename R>
  inline double dist(T&& x, R&& z) const;

  template<typename T, typename R>
  inline double Q_func(T&& a, R&& b) const;

  template<typename T>
  inline bool allContains(T&& t) const;

  template<typename T>
  inline void assignSet(T&& t);

  inline void reset();

  inline std::list<std::string> getNames() const;

private:
  std::vector<Kernel<StoreDots>> kernels_;
  std::vector<double> weights_;
  int size_;
};

inline Combination::Combination(): size_(0)
{
}

inline int Combination::size() const
{
  return size_;
}

inline void Combination::addKernel(std::unique_ptr<KernelFile>&& file, Kernel<StoreDots>::kernel_type type, double gamma, double avg)
{
  kernels_.emplace_back(Kernel<StoreDots>(std::move(file), type, avg, gamma));
  weights_.push_back(0);
  ++size_;
}

inline auto& Combination::getKernels()
{
  return kernels_;
}

inline const auto& Combination::getKernels() const
{
  return kernels_;
}

inline const auto& Combination::getKernel(int n) const
{
  return kernels_[n];
}

inline auto& Combination::getKernel(int n)
{
  return kernels_[n];
}

inline auto& Combination::getWeights()
{
  return weights_;
}

inline const auto& Combination::getWeights() const
{
  return weights_;
}

inline double Combination::getWeight(int n) const
{
  return weights_[n];
}

template<typename T, typename R>
inline double Combination::k_func(T&& a, R&& b) const
{
  double val = 0.0;
  for(int n = 0; n < size_; n++)
    val += kernels_[n].k_func(a,b) * weights_[n];
  return val;
}

template<typename T, typename R>
inline double Combination::dist(T&& x, R&& z) const
{
  double val = 0.0;
  for(int n = 0; n < size_; n++)
    val += kernels_[n].dist(x,z);
  return val;
}

template<typename T, typename R>
inline double Combination::Q_func(T&& a, R&& b) const
{
  double val = 0.0;
  for(int n = 0; n < size_; n++)
    val += kernels_[n].Q_func(a,b) * weights_[n];
  return val;
}

template<typename T>
inline bool Combination::allContains(T&& t) const
{
  for(const auto& k : kernels_)
  {
    if(!k.contains(t)) return false;
  }
  return true;
};

inline void Combination::reset()
{
  kernels_.clear();
  weights_.clear();
}

inline std::list<std::string> Combination::getNames() const
{
  std::list<std::string> names;
  for(const auto& k : kernels_)
    names.merge(k.getNames(), [](const auto& a, const auto& b)
  {
    return a < b;
  });
  names.unique();
  return names;
}

template<typename T>
inline void Combination::assignSet(T&& t)
{
  for(auto& k:kernels_)
    k.assignSet(t);
}

#endif // COMBINATION_H
