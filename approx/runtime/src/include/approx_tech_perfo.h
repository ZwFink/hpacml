#ifndef __APPROX_TECH_PERFO__
#define __APPROX_TECH_PERFO__

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <approx_data.h>
#include <approx_tech_base.h>

class ApproximateTechniquePerfo : public ApproximateTechniqueBase {

public:
  ApproximateTechniquePerfo(std::string Name)
      : ApproximateTechniqueBase(Name){};
  ~ApproximateTechniquePerfo(){};

  virtual void Approximate(std::shared_ptr<ApproxInstance> &instance) {
    std::cout << approxName << "Approximate Not implemented\n";
    std::cout << "Executing Accurate\n";
    instance->accurateFn(instance->argsFn);
  }
};

#endif