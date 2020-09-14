#ifndef __APPROX_TECH_PROFILE__
#define __APPROX_TECH_PROFILE__

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <approx_data.h>
#include <approx_tech_base.h>

class ApproximateTechniqueProfile : public ApproximateTechniqueBase {
  /// The profiling approximation technique is simple. We record everything
  /// annotated by the user, when the application execution is done we will
  /// write everything on stdout. We should extern this behavior to write into
  /// files Easily processed by python.
  std::unordered_map<std::string, std::vector<std::shared_ptr<ApproxInstance>>>
      appHistory;

public:
  ApproximateTechniqueProfile(std::string Name)
      : ApproximateTechniqueBase(Name) {}
  ~ApproximateTechniqueProfile();

  // Here process any input data required by this approximation technique
  virtual void PreProcess(std::shared_ptr<ApproxInstance> &instance);

  virtual void Approximate(std::shared_ptr<ApproxInstance> &instance) {
    instance->accurateFn(instance->argsFn);
  };

  /// Here process any output data required by this approximation technique
  virtual void PostProcess(std::shared_ptr<ApproxInstance> &instance);
};

#endif