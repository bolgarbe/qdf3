#include "file_handler.h"

#include <fstream>
#include <cstdlib>
#include <cstring>
#include <iostream>

KernelFile::KernelFile(const char* file)
{
  open(file);
}

void KernelFile::open(const char* file)
{
  std::ifstream in(file);
  if(!in.is_open())
  {
    std::ostringstream s;
    s << "Failed to open file: " << file;
    throw Exception(s.str().c_str());
  }

  samples_ = 0;
  max_index_ = 0;

  std::string line;

  std::vector<Eigen::Triplet<double>> triplets;
  while(getline(in, line))
  {
    std::string::size_type pre_space = line.find_first_of(" ") + 1;
    std::string name(line.substr(0, pre_space - 1));

    if(name.empty())
    {
      std::ostringstream s;
      s << "Error: Empty ID at position "<<samples_<<"in file "<<file;
      throw Exception(s.str().c_str());
    }

    if(positions_.count(name)>0)
    {
      std::ostringstream s;
      s << "Error: Duplicate ID "<<name<<" at "<<samples_<<" in file "<<file;
      throw Exception(s.str().c_str());
    }

    positions_.insert(std::make_pair(name, samples_));
    std::string::size_type colon = line.find_first_of(":");
    std::string::size_type post_space = line.find_first_of(" ", colon);
    int index = 0;
    while(colon != std::string::npos)
    {
      index = atoi(line.substr(pre_space, colon - pre_space).c_str());
      std::string tmp(line.substr(colon + 1, post_space - colon));
      double value = atof(tmp.c_str());
      max_index_ = index > max_index_ ? index : max_index_;

      triplets.push_back(Eigen::Triplet<double>(samples_, index, value));

      colon = line.find_first_of(":", post_space);
      pre_space = line.find_first_of(" ", post_space) + 1;
      post_space = line.find_first_of(" ", colon);
    }
    ++samples_;
  }
  data_.resize(samples_, max_index_ + 1);
  data_.setFromTriplets(triplets.begin(), triplets.end());
}
