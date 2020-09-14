#ifndef __APPROX_TECH_BASE__
#define __APPROX_TECH_BASE__

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <approx_data.h>

class ApproximateTechniqueBase {
protected:
  std::string approxName;

public:
  ApproximateTechniqueBase(std::string Name) : approxName(Name){};
  virtual ~ApproximateTechniqueBase(){};

  // Here process any input data required by this approximation technique
  /// The Base class is not doing anything during pre process.
  virtual void PreProcess(std::shared_ptr<ApproxInstance> &instance) {
    std::cout << approxName << " Does not require Pre-Process\n";
  };

  /// Perform approximation.
  /// At the end the approximated value of this technique should be
  /// already copied/moved to the application memory space.
  virtual void Approximate(std::shared_ptr<ApproxInstance> &instance) = 0;

  /// Accurate Execution of the implementation
  void Accurate(std::shared_ptr<ApproxInstance> &instance) {
    instance->accurateFn(instance->argsFn);
  };

  /// Here process any output data required by this approximation technique
  /// The Base class is not doing anything
  virtual void PostProcess(std::shared_ptr<ApproxInstance> &instance) {
    std::cout << approxName << " Does not require PostProcess\n";
  };

  /// Execution step of the application
  void Execute(std::shared_ptr<ApproxInstance> &instance) {
    if (instance->cond) {
      this->Approximate(instance);
    } else {
      Accurate(instance);
    }
    return;
  }
};

#endif