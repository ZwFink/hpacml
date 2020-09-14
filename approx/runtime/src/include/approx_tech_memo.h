#ifndef __APPROX_TECH_MEMO__
#define __APPROX_TECH_MEMO__

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <approx_data.h>
#include <approx_tech_base.h>


class ApproximateTechniqueMemoIn : public ApproximateTechniqueBase {
public:
  ApproximateTechniqueMemoIn(std::string Name)
      : ApproximateTechniqueBase(Name){};
  ~ApproximateTechniqueMemoIn(){};

  /// Memoization Will need some PreProcessing
  virtual void PreProcess(std::shared_ptr<ApproxInstance> &instance) {
    std::cout << approxName << " Pre-Process not implemented\n";
  }

  /// Memo-In approximation.
  /// At the end the approximated value of this technique should be
  /// already copied/moved to the application memory space.
  virtual void Approximate(std::shared_ptr<ApproxInstance> &instance) {
    std::cout << approxName << " Approximation Is not implemented yet\n";
    instance->accurateFn(instance->argsFn);
  };

  /// Here process any output data required
  /// by this approximation technique
  /// The Base class is not doing anything
  /// Memoization will need some PostProcessing.
  /// For example moving data from the library
  /// memory to the memory of the application
  virtual void PostProcess(std::shared_ptr<ApproxInstance> &instance) {
    std::cout << approxName << " Post-Process not implemented\n";
  }
};

class ApproximateTechniqueMemoOut : public ApproximateTechniqueBase {
public:
  ApproximateTechniqueMemoOut(std::string Name)
      : ApproximateTechniqueBase(Name) {}
  ~ApproximateTechniqueMemoOut() {}

  virtual void PreProcess(std::shared_ptr<ApproxInstance> &instance) {
    std::cout << approxName << " Pre-Process not implemented\n";
  }

  /// MemoOut approximation.
  /// At the end the approximated value of this technique should be
  /// already copied/moved to the application memory space.
  virtual void Approximate(std::shared_ptr<ApproxInstance> &instance) {
    std::cout << approxName << " Approximation Is not implemented yet\n";
    instance->accurateFn(instance->argsFn);
  }

  /// Here process any output data required
  /// by this approximation technique
  /// The Base class is not doing anything
  /// Memoization will need some PostProcessing.
  /// For example moving data from the library
  /// memory to the memory of the application
  virtual void PostProcess(std::shared_ptr<ApproxInstance> &instance) {
    std::cout << approxName << " Post-Process not implemented\n";
  }
};

#endif
