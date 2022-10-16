#ifndef FILE_HANDLER_H
#define FILE_HANDLER_H

#include "Eigen/SparseCore"
#include "exceptions.h"

#include <string>
#include <map>
#include <list>

template<typename FP>
class HandlerBase
{
public:
  HandlerBase();

  virtual ~HandlerBase() = default;

  inline auto getNames() const;

  inline int getSamplesNum() const;

  inline const auto& getData() const;

  inline const auto& getPositions() const;

  template<typename T>
  inline int getPos(T&& q) const;

  template<typename T>
  inline bool contains(T&& q) const;

protected:
  int samples_;
  int max_index_;

  Eigen::SparseMatrix<double> data_;
  std::map<std::string, int> positions_;
};

class KernelFile: public HandlerBase<KernelFile>
{
public:
  KernelFile() = default;

  KernelFile(const char* file);

  void open(const char* file);
};

template<typename FP>
HandlerBase<FP>::HandlerBase():samples_(0),max_index_(0)
{
}

template<typename FP>
inline auto HandlerBase<FP>::getNames() const
{
  std::list<std::string> names;
  for(const auto& i : positions_)
    names.push_back(i.first);
  return names;
}

template<typename FP>
inline int HandlerBase<FP>::getSamplesNum() const
{
  return samples_;
}

template<typename FP>
inline const auto& HandlerBase<FP>::getData() const
{
  return data_;
}

template<typename FP>
inline const auto& HandlerBase<FP>::getPositions() const
{
  return positions_;
}

template<typename FP>
template<typename T>
inline int HandlerBase<FP>::getPos(T&& q) const
{
  return positions_.at(std::forward<T>(q));
}

template<typename FP>
template<typename T>
inline bool HandlerBase<FP>::contains(T&& q) const
{
  return positions_.count(std::forward<T>(q))>0;
}

class PrecomputedFile: public HandlerBase<PrecomputedFile>
{
public:
  void open(const char* file)
  {
    // not implemented yet
  }
};

#endif //FILE_HANDLER_H
