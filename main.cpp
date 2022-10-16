#include "solver.h"
#include "sigmoid.h"

#include <iostream>
#include <fstream>
#include <memory>
#include <list>

void usage(const char* executable)
{
  std::cout << "Usage: "
            << "\n   -c MKL parameter C list (e.g. 1,10,100,1000, default: 100)"
            << "\n   -e MKL parameter epsilon (default: 0.0001)"
            << "\n   -l MKL parameter lambda (default: 1)"
            << "\n   -f Output file prefix (default: test)"
            << "\n   -i cv iterations (default: 1, don't do cv)"
            << "\n   -k kernel file"
            << "\n   -t training file (queries)"
            << "\n   -r do probability estimates"
            << "\nKernel file format:"
            << "\n   path_to_kernel_1 type_1 param_11,param12,param13,..."
            << "\n   path_to_kernel_2 type_2 param_21,param22,param23,..."
            << "\n   ..."
            << "\nTypes:"
            << "\n   0=cosine, 1=Tanimoto, 2=RBF, 3=linear, 4=quadratic, 5=polynomial"
            << "\nTraining file format:"
            << "\n   +q1entity1,+q1entity2,...;-q1entity1,-q1entity2,..."
            << "\n   +q2entity1,+q2entity2,...;-q2entity1,-q2entity2,..."
            << "\n\nExample: " << executable << " -k mykernels.txt -t myqueries.txt -f pr1\n"
            << std::endl;
}

template<typename T>
class Singleton
{
  Singleton(const Singleton&) = delete;
  Singleton& operator=(const Singleton&) = delete;

protected:
    Singleton() = default;
    virtual ~Singleton() = default;

public:
  template<typename... A>
  static T& getInst(A&&... a)
  {
    static T inst_(std::forward<A>(a)...);
    return inst_;
  }
};

class OutputHandler:public Singleton<OutputHandler>
{
public:
  enum class Channel {SC,ENT,WG,AL};

  OutputHandler(const char* file_prefix):
    sc_(std::string(file_prefix) + "_scores"),
    ec_(std::string(file_prefix) + "_entities"),
    wc_(std::string(file_prefix) + "_weights"),
    ac_(std::string(file_prefix) + "_alphas"),
    sf_(""),ef_(""),wf_(""),af_("")
  {

  }

  ~OutputHandler()
  {
    sc_.close();
    ec_.close();
    wc_.close();
    ac_.close();
  }

  template<typename T>
  void write(T&& s,Channel c)
  {
    switch(c)
    {
    case Channel::SC:
      sc_<<sf_<<s; sf_=",";
      break;
    case Channel::ENT:
      ec_<<ef_<<s; ef_=",";
      break;
    case Channel::WG:
      wc_<<wf_<<s; wf_=",";
      break;
    case Channel::AL:
      ac_<<af_<<s; af_=",";
    }
  }

  void newLine()
  {
    sc_<<std::endl; sf_="";
    ec_<<std::endl; ef_="";
    wc_<<std::endl; wf_="";
    ac_<<std::endl; af_="";
  }

private:
  std::ofstream sc_,ec_,wc_,ac_;
  std::string sf_,ef_,wf_,af_;
};

/*template<typename T,typename M>
void do_prediction(const char* file_prefix, T&& all_samples, M&& model)
{
  auto&& inst = OutputHandler::getInst(file_prefix);

  for(auto w:model.getWeights())
    inst.write(w,OutputHandler::Channel::WG);

  std::multimap<double, std::string> res;
  for(const auto& smp : all_samples)
  {
    res.insert(std::make_pair(model.score(smp),smp));
  }

  std::ofstream test(file_prefix);
  for(auto r=res.crbegin(); r!=res.rend(); ++r)
  {
    test<<r->second<<" "<<r->first<<std::endl;
    inst.write(r->first,OutputHandler::Channel::SC);
    inst.write(r->second,OutputHandler::Channel::ENT);
  }
  inst.newLine();

  test.close();
}*/

template<typename T,typename M>
void do_prediction(const char* file_prefix, T&& all_samples, M&& model)
{
  auto&& inst = OutputHandler::getInst(file_prefix);

  for(auto w:model.getWeights())
    inst.write(w,OutputHandler::Channel::WG);

  for(auto a:model.getAlphas())
    inst.write(a,OutputHandler::Channel::AL);

  auto scores = model.scores(all_samples);

  std::multimap<double, std::string> res;
  int i=0;
  for(const auto& smp : all_samples)
  {
    res.insert(std::make_pair(scores(i),smp));
    ++i;
  }

  //std::ofstream test(file_prefix);
  for(auto r=res.crbegin(); r!=res.rend(); ++r)
  {
    //test<<r->second<<" "<<r->first<<std::endl;
    inst.write(r->first,OutputHandler::Channel::SC);
    inst.write(r->second,OutputHandler::Channel::ENT);
  }
  inst.newLine();

  //test.close();
}

template<typename L, typename T,typename M>
void do_prob(const char* file_prefix, L&& training_samples, T&& all_samples, M&& model)
{
  Sigmoid s;
  s.train(std::forward<L>(training_samples), model);

  auto&& inst = OutputHandler::getInst(file_prefix);

  for(auto w:model.getWeights())
    inst.write(w,OutputHandler::Channel::WG);

  for(auto a:model.getAlphas())
    inst.write(a,OutputHandler::Channel::AL);

  auto scores = model.scores(all_samples);
  std::multimap<double, std::string> res;
  int i=0;
  for(const auto& smp : all_samples)
  {
    res.insert(std::make_pair(s.probEstimate(scores[i++]),smp));
  }

  //std::ofstream test(file_prefix);
  for(auto r=res.crbegin(); r!=res.rend(); ++r)
  {
    //test<<r->second<<" "<<r->first<<std::endl;
    inst.write(r->first,OutputHandler::Channel::SC);
    inst.write(r->second,OutputHandler::Channel::ENT);
  }
  inst.newLine();

  //test.close();
}

template<typename C, typename T, typename R>
double do_crossval(SolverParams params, C&& Q, T&& training_samples, R&& all_samples, int fold=100)
{
  int p_s=0,n_s=0;
  int all_s=all_samples.size();
  std::vector<std::string> positive_samples, negative_samples;

  for(auto&& i:training_samples)
  {
    if(i.y==1) {positive_samples.push_back(i.x); ++p_s;}
    else {negative_samples.push_back(i.x); ++n_s;}
  }

  int p_t=p_s*0.7;
  int n_t=n_s*0.7;

  double auc=0;
  for(int i=0; i<fold; i++)
  {
    std::random_shuffle(positive_samples.begin(),positive_samples.end());
    std::random_shuffle(negative_samples.begin(),negative_samples.end());

    std::vector<MklTrainingSample> tr;
    for(int j=0; j<p_t; j++) tr.emplace_back(MklTrainingSample(positive_samples[j],1));
    for(int j=0; j<n_t; j++) tr.emplace_back(MklTrainingSample(negative_samples[j],-1));
    TwoClassSolver s;

    auto model = s.solve(params,Q,tr);

    int smp_num = all_samples.size();
    /*std::vector<double> scores(smp_num);
    int ctr=0;
    for(auto s: all_samples)
    {
      scores[ctr++]=model.score(s);
    }*/
    auto scores = model.scores(all_samples);

    double correct=0.0, all=(p_s-p_t)*all_s;
    for(int j=p_t; j<p_s; j++)
    {
      const auto& smp=positive_samples[j];
      double sc=model.score(smp);
      for(int s=0; s<smp_num; s++)
      {
        if(sc>scores[s]) ++correct;
      }
    }
    auc+=correct/all;
  }

  return auc/fold;
}

struct GridParams
{
  GridParams(int num_kernels):kernel_params(num_kernels,0),c(0),auc(-1){}

  std::vector<double> kernel_params;
  double c,auc;
};

template<typename C, typename T, typename R, typename KP, typename CP>
GridParams recursive_crossval(SolverParams params,
                          C&& Q,
                          T&& training_samples,
                          R&& all_samples,
                          int fold,
                          KP&& kernel_params,
                          CP&& c_params,
                          int num_kernels,
                          int pr_num)
{
  GridParams best_gp(num_kernels);

  if(pr_num!=num_kernels-1)
  {
    for(auto i:kernel_params[pr_num])
    {
      Q.getKernel(pr_num).setParam(i);
      std::cout<<"Setting param for kernel "<<pr_num<<" to "<<i<<std::endl;
      auto gp = recursive_crossval(params,Q,training_samples,all_samples,fold,kernel_params,c_params,num_kernels,pr_num+1);

      if(gp.auc>best_gp.auc)
      {
        best_gp=gp;
        best_gp.kernel_params[pr_num]=i;
      }
    }
  }
  else
  {
    double best_outer_auc = -1;
    for(auto i:kernel_params[pr_num])
    {
      Q.getKernel(pr_num).setParam(i);
      std::cout<<"Setting param for kernel "<<pr_num<<" to "<<i<<std::endl;

      double best_inner_auc = -1;
      double best_c;
      for(auto j:c_params)
      {
        params.c=j;
        double inner_auc=do_crossval(params,Q,training_samples,all_samples,fold);
        std::cout<<"Evaluated for c="<<j<<", got auc "<<inner_auc<<std::endl;
        if(inner_auc>best_inner_auc)
        {
          best_c=j;
          best_inner_auc=inner_auc;
        }
      }

      if(best_inner_auc>best_outer_auc)
      {
        best_gp.kernel_params[pr_num]=i;
        best_gp.c=best_c;
        best_outer_auc=best_inner_auc;
      }
    }
    best_gp.auc=best_outer_auc;
  }

  return best_gp;
}

template<typename T>
auto get_params(T&& param_list)
{
  std::vector<double> param_vec;
  std::string::size_type pos=param_list.find_first_of(","),lpos=0;
  while(pos!=std::string::npos)
  {
    param_vec.push_back(atof(param_list.substr(lpos,pos-lpos).c_str()));
    lpos=pos+1;
    pos=param_list.find_first_of(",",lpos);
  }
  param_vec.push_back(atof(param_list.substr(lpos,pos-lpos).c_str()));
  return param_vec;
}

int main(int argc, char** argv)
{
  srand(time(NULL));
  std::ifstream kernel_file;
  std::ifstream label_file;
  std::ifstream training_file;
  const char* file_prefix = "test";
  std::vector<double> c_list = {100};
  int fold=1;
  bool train_log_reg=false;

  SolverParams params;
  params.eps = 1e-4;
  params.lambda = 1;
  params.nu = 0.4;
  params.c = 100;
  params.r_eps = 0.1;

  // parse command line
  if(argc < 2)
  {
    usage(argv[0]);
    return 0;
  }
  int a = 1;
  while(a < argc)
  {
    switch(argv[a][1])
    {
    case 'h':
      usage(argv[0]);
      return 0;
    case 'k':
      kernel_file.open(argv[++a]);
      break;
    case 'n':
      params.nu = atof(argv[++a]);
      break;
    case 'c':
      c_list = get_params(std::string(argv[++a]));
      break;
    case 'i':
      fold = atoi(argv[++a]);
      break;
    case 'e':
      params.eps = atof(argv[++a]);
      break;
    case 'l':
      params.lambda = atof(argv[++a]);
      break;
    case 't':
      training_file.open(argv[++a]);
      break;
    case 'p':
      params.r_eps = atof(argv[++a]);
      break;
    case 'y':
      label_file.open(argv[++a]);
      break;
    case 'f':
      file_prefix = argv[++a];
      break;
    case 'r':
      train_log_reg = true;
      break;
    default:
      usage(argv[0]);
      exit(0);
      break;
    }
    ++a;
  }

  if(!kernel_file.is_open())
  {
    std::cout << "Can't open kernel file!" << std::endl;
    return -1;
  }
  if(!training_file.is_open())
  {
    std::cout << "Can't open training file!" << std::endl;
    return -1;
  }

  std::vector<std::vector<double>> k_param_list;

  Combination Q;
  {
    std::string file_line;
    while(getline(kernel_file, file_line))
    {
      if(!file_line.empty())
      {
        std::istringstream str(file_line);
        std::string file_name,param_list;
        int kernel_type;

        str >> file_name >> kernel_type >> param_list;

        k_param_list.emplace_back(get_params(param_list));

        switch(kernel_type)
        {
        case 0:
          Q.addKernel(std::make_unique<KernelFile>(file_name.c_str()), Kernel<StoreDots>::kernel_type::COSINE);
          break;
        case 1:
          Q.addKernel(std::make_unique<KernelFile>(file_name.c_str()), Kernel<StoreDots>::kernel_type::TANIMOTO);
          break;
        case 2:
          Q.addKernel(std::make_unique<KernelFile>(file_name.c_str()), Kernel<StoreDots>::kernel_type::RBF);
          break;
        case 3:
          Q.addKernel(std::make_unique<KernelFile>(file_name.c_str()), Kernel<StoreDots>::kernel_type::LINEAR);
          break;
        case 4:
          Q.addKernel(std::make_unique<KernelFile>(file_name.c_str()), Kernel<StoreDots>::kernel_type::QUADRATIC);
          break;
        case 5:
          Q.addKernel(std::make_unique<KernelFile>(file_name.c_str()), Kernel<StoreDots>::kernel_type::POLY);
          break;
        }
      }
    }
    kernel_file.close();
  }

  auto all_samples = std::move(Q.getNames());

  std::string line;
  while(getline(training_file, line))
  {
    if(!line.empty())
    {
      double y = 1.0;
      std::vector<MklTrainingSample> training_samples;
      std::vector<std::string> training_names;
      std::string::size_type pos = line.find_first_of(",;");
      std::string::size_type last_pos = 0;
      while(pos != std::string::npos)
      {
        training_samples.emplace_back(MklTrainingSample(line.substr(last_pos, pos - last_pos), y));
        training_names.emplace_back(line.substr(last_pos, pos - last_pos));
        if(line[pos] == ';')
        {
          y = -1.0;
        }
        last_pos = pos + 1;
        pos = line.find_first_of(",;", last_pos);
      }
      training_samples.emplace_back(MklTrainingSample(line.substr(last_pos, pos - last_pos), y));
      training_names.emplace_back(line.substr(last_pos, pos - last_pos));
      Q.assignSet(training_names);

      if(!training_samples.empty())
      {
        int num_kernels = k_param_list.size();
        if(fold>1)
        {
          auto gp = recursive_crossval(params,Q,training_samples,all_samples,fold,k_param_list,c_list,num_kernels,0);

          std::cout<<"Overall best auc: "<<gp.auc<<" with params:"<<std::endl;

          for(int i=0; i<num_kernels; i++)
          {
            std::cout<<"Kernel "<<i<<": "<<gp.kernel_params[i]<<std::endl;
          }
          std::cout<<"C: "<<gp.c<<std::endl;

          params.c=gp.c;
          for(int i=0; i<num_kernels; i++)
          {
            Q.getKernel(i).setParam(gp.kernel_params[i]);
          }
        }
        else
        {
          params.c = c_list[0];
          for(int i=0; i<num_kernels; i++)
          {
            Q.getKernel(i).setParam(k_param_list[i][0]);
          }
        }

        TwoClassSolver s;
        auto model = s.solve(params,Q,training_samples);

        if(!train_log_reg)
          do_prediction(file_prefix,all_samples,model);
        else
          do_prob(file_prefix,training_samples,all_samples,model);

        model.scores(all_samples);
      }
    }
  }

  training_file.close();

  return 0;
}
