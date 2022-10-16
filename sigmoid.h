#ifndef SIGMOID_H
#define SIGMOID_H

#include <algorithm>
#include <iostream>

class Sigmoid
{
public:
  Sigmoid();

  template<typename T, typename M>
  void train(T&& smp, M&& model);

  inline double probEstimate(double score) const;
  inline double getA() const;
  inline double getB() const;

private:
  double a_,b_;
};


/* From LibSVM */
template<typename T, typename M>
void Sigmoid::train(T&& smp, M&& model)
{
  int num_all = smp.size();
  double num_pos = std::count_if(smp.begin(), smp.end(), [](const auto& a){return a.y==1;});
  double num_neg = num_all-num_pos;

  std::vector<double> scores(num_all), t(num_all);

  a_=0.0;
  b_=log((num_neg+1.0)/(num_pos+1.0));
  double f=0.0;

  double hi=(num_pos+1.0)/(num_pos+2.0);
  double lo=1.0/(num_neg+2.0);

  for(int i=0; i<num_all; i++)
  {
    double si=model.score(smp[i].x);
    double ti=smp[i].y > 0.0 ? hi : lo;
    scores[i]=si;
    t[i]=ti;

    double fApB=si*a_+b_;
    f += fApB >= 0.0 ?
      ti*fApB+log(1.0+exp(-fApB)) :
      (ti-1.0)*fApB+log(1.0+exp(fApB));
  }

  while(true)
  {
    double h11=1e-12,h22=1e-12;
    double h21=0.0,g1=0.0,g2=0.0;

    for(int i=0; i<num_all; i++)
    {
      double fApB = scores[i]*a_+b_;
      double p = fApB >= 0.0 ?
        exp(-fApB)/(1.0+exp(-fApB)) :
        1.0/(1.0+exp(fApB));
      double q = fApB >= 0.0 ?
        1.0/(1.0+exp(-fApB)) :
        exp(fApB)/(1.0+exp(fApB));

      double d2=p*q;
      h11 += scores[i]*scores[i]*d2;
      h22 += d2;
      h21 += scores[i]*d2;

      double d1=t[i]-p;
      g1+=scores[i]*d1;
      g2+=d1;
    }

    if(fabs(g1)<1e-5 && fabs(g2)<1e-5) break;

    double det=h11*h22-h21*h21;
    double dA=-(h22*g1-h21*g2)/det;
    double dB=-(-h21*g1+h11*g2)/det;
    double gd=g1*dA+g2*dB;

    double step=1.0;
    while(step>1e-10)
    {
      double newA=a_+step*dA;
      double newB=b_+step*dB;
      double newf=0.0;

      for(int i=0; i<num_all; i++)
      {
        double fApB=scores[i]*newA+newB;
        newf+= fApB>=0.0 ?
          t[i]*fApB+log(1.0+exp(-fApB)) :
          (t[i]-1.0)*fApB+log(1.0+exp(fApB));
      }

      if(newf<f+0.0001*step*gd)
      {
        a_=newA; b_=newB; f=newf;
        break;
      }
      else
      {
        step/=2.0;
      }
    }

    if(step<1e-10)
    {
      std::cout<<"Line search failed"<<std::endl;
      break;
    }
  }
}

inline double Sigmoid::probEstimate(double score) const
{
  return 1.0/(1.0+exp(score*a_+b_));
}

inline double Sigmoid::getA() const
{
  return a_;
}

inline double Sigmoid::getB() const
{
  return b_;
}

#endif //SIGMOID_H
