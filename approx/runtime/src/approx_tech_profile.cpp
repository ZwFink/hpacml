#include <approx_tech_profile.h>

void ApproximateTechniqueProfile::PreProcess(std::shared_ptr<ApproxInstance> &instance) {
  appHistory[instance->getName()].emplace_back(instance);
  instance->registerInputs();
}

void ApproximateTechniqueProfile::PostProcess(std::shared_ptr<ApproxInstance> &instance) {
  instance->registerOutputs();
}

ApproximateTechniqueProfile::~ApproximateTechniqueProfile() {
  std::cout <<"--------------------- Recoreded Input/Output Values -----------------\n";
  for (auto v : appHistory) {
    std::cout << v.first << " " << v.second.size() << std::endl;
    for (auto i : v.second)
      i->dump();
  }
  std::cout <<"--------------------- Finished Input/Output Values ------------------\n";
}

