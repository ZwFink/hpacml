#include <iostream>
#include <memory>
#include <stack>
#include <stdint.h>
#include <unordered_map>

#include <approx_data.h>
#include <approx_tech_base.h>
#include <approx_tech_memo.h>
#include <approx_tech_perfo.h>
#include <approx_tech_profile.h>
#include <approx_types.h>

/// This integration between the compiler and the RT is quite thin.
/// I will add an issue to fix this.
enum ApproximationTechniques { Perforation = 0, MemoIn, MemoOut, Profile };

class ApproximateRT {
public:
  /// For each approximate annotated region track how many times
  /// it was invoked.
  std::unordered_map<std::string, unsigned long> appInvocations;
  /// A stack storing the current active approximations.
  /// This will be usefull for accessing nested approximation techniques.
  /// as we always need to peak at the top of the stack.
  std::stack<std::pair<std::shared_ptr<ApproximateTechniqueBase>,
                       std::shared_ptr<ApproxInstance>>>
      AppRegion;
  /// Pointers to all implemented approximation techniques.
  std::vector<std::shared_ptr<ApproximateTechniqueBase>> approxTechniques;

  ApproximateRT();
  ~ApproximateRT(){};
};

ApproximateRT::ApproximateRT() {
  approxTechniques.emplace_back(
      std::make_shared<ApproximateTechniquePerfo>(std::string("Perforation")));
  approxTechniques.emplace_back(
      std::make_shared<ApproximateTechniqueMemoIn>(std::string("Memo-in")));
  approxTechniques.emplace_back(
      std::make_shared<ApproximateTechniqueMemoOut>(std::string("Memo-out")));
  approxTechniques.emplace_back(
      std::make_shared<ApproximateTechniqueProfile>(std::string("Profile")));
}

ApproximateRT approxRT;

void fillInstance(std::shared_ptr<ApproxInstance> &Instance,
                  approx_var_info_t *vars, int num_deps) {
  TypeInterface *dataValue;
  for (int i = 0; i < num_deps; i++) {
    switch (vars[i].data_type) {
    case SChar:
    case Char_U:
      if (vars[i].dir == Input || vars[i].dir == InputOutput) {
        dataValue = new Type<char>(vars[i].ptr, (MType)vars[i].data_type,
                                   vars[i].num_elem, vars[i].sz_elem,
                                   vars[i].sz_bytes, i);
        Instance->addValue(dataValue, Input);
      }
      if (vars[i].dir == Output || vars[i].dir == InputOutput) {
        dataValue = new Type<char>(vars[i].ptr, (MType)vars[i].data_type,
                                   vars[i].num_elem, vars[i].sz_elem,
                                   vars[i].sz_bytes, i);
        Instance->addValue(dataValue, Output);
      }
      break;
    case UInt:
      if (vars[i].dir == Input || vars[i].dir == InputOutput) {
        dataValue = new Type<unsigned int>(
            vars[i].ptr, (MType)vars[i].data_type, vars[i].num_elem,
            vars[i].sz_elem, vars[i].sz_bytes, i);
        Instance->addValue(dataValue, Input);
      }
      if (vars[i].dir == Output || vars[i].dir == InputOutput) {
        dataValue = new Type<unsigned int>(
            vars[i].ptr, (MType)vars[i].data_type, vars[i].num_elem,
            vars[i].sz_elem, vars[i].sz_bytes, i);
        Instance->addValue(dataValue, Output);
      }
      break;
    case Int:
      if (vars[i].dir == Input || vars[i].dir == InputOutput) {
        dataValue = new Type<int>(vars[i].ptr, (MType)vars[i].data_type,
                                  vars[i].num_elem, vars[i].sz_elem,
                                  vars[i].sz_bytes, i);
        Instance->addValue(dataValue, Input);
      }
      if (vars[i].dir == Output || vars[i].dir == InputOutput) {
        dataValue = new Type<int>(vars[i].ptr, (MType)vars[i].data_type,
                                  vars[i].num_elem, vars[i].sz_elem,
                                  vars[i].sz_bytes, i);
        Instance->addValue(dataValue, Output);
      }
      break;
    case Double:
      if (vars[i].dir == Input || vars[i].dir == InputOutput) {
        dataValue = new Type<double>(vars[i].ptr, (MType)vars[i].data_type,
                                     vars[i].num_elem, vars[i].sz_elem,
                                     vars[i].sz_bytes, i);
        Instance->addValue(dataValue, Input);
      }
      if (vars[i].dir == Output || vars[i].dir == InputOutput) {
        dataValue = new Type<double>(vars[i].ptr, (MType)vars[i].data_type,
                                     vars[i].num_elem, vars[i].sz_elem,
                                     vars[i].sz_bytes, i);
        Instance->addValue(dataValue, Output);
      }
      break;
    case Float:
      if (vars[i].dir == Input || vars[i].dir == InputOutput) {
        dataValue = new Type<float>(vars[i].ptr, (MType)vars[i].data_type,
                                    vars[i].num_elem, vars[i].sz_elem,
                                    vars[i].sz_bytes, i);
        Instance->addValue(dataValue, Input);
      }
      if (vars[i].dir == Output || vars[i].dir == InputOutput) {
        dataValue = new Type<float>(vars[i].ptr, (MType)vars[i].data_type,
                                    vars[i].num_elem, vars[i].sz_elem,
                                    vars[i].sz_bytes, i);
        Instance->addValue(dataValue, Output);
      }
      break;
    default:
      break;
    }
  }
}

void registerInstance(std::string name, void (*accFn)(void *),
                      void (*perfoFn)(void *), void *arg, bool cond,
                      approx_perfo_info_t *perfoArgs, int memo_type,
                      approx_var_info_t *vars, int num_deps) {
  std::shared_ptr<ApproxInstance> Instance = std::make_shared<ApproxInstance>(
      name, approxRT.appInvocations[name]++, accFn, perfoFn, arg, cond,
      perfoArgs, memo_type);
  fillInstance(Instance, vars, num_deps);
  if (perfoFn != nullptr) {
    approxRT.AppRegion.push(
        std::make_pair(approxRT.approxTechniques[Perforation], Instance));
  } else if (memo_type == MemoIn) {
    approxRT.AppRegion.push(
        std::make_pair(approxRT.approxTechniques[MemoIn], Instance));
  } else if (memo_type == MemoOut) {
    approxRT.AppRegion.push(
        std::make_pair(approxRT.approxTechniques[MemoOut], Instance));
  } else {
    approxRT.AppRegion.push(
        std::make_pair(approxRT.approxTechniques[Profile], Instance));
  }
}

void PreProcess() {
  std::pair<std::shared_ptr<ApproximateTechniqueBase>,
            std::shared_ptr<ApproxInstance>>
      tmp = approxRT.AppRegion.top();
  tmp.first->PreProcess(tmp.second);
}

void PostProcess() {
  std::pair<std::shared_ptr<ApproximateTechniqueBase>,
            std::shared_ptr<ApproxInstance>>
      tmp = approxRT.AppRegion.top();
  tmp.first->PostProcess(tmp.second);
}

void Execute() {
  std::pair<std::shared_ptr<ApproximateTechniqueBase>,
            std::shared_ptr<ApproxInstance>>
      tmp = approxRT.AppRegion.top();
  tmp.first->Execute(tmp.second);
}

void unRegisterInstance() {
  if (!approxRT.AppRegion.empty()) {
    approxRT.AppRegion.top().second->unRegisterInstance();
    approxRT.AppRegion.pop();
  }
}