#include <stdint.h>
#include <stdio.h>

#include <approx.h>
#include <approx_data.h>
#include <approx_types.h>

void registerInstance(std::string name, void (*accFn)(void *),
                      void (*perfoFn)(void *), void *arg, bool cond,
                      approx_perfo_info_t *perfoArgs, int memo_type,
                      approx_var_info_t *vars, int num_deps);
void unRegisterInstance();
void PreProcess();
void Execute();
void PostProcess();

void __approx_exec_call(void (*accFn)(void *), void (*perfoFn)(void *),
                        void *arg, bool cond, void *perfoArgs, void *deps,
                        int num_deps, int memo_type) {
  approx_perfo_info_t *perfo = (approx_perfo_info_t *)perfoArgs;
  approx_var_info_t *var_info = (approx_var_info_t *)deps;
  /// I am not sure that it is conceptually correct whether we should
  /// do anything when condition is false. This is a point of discussion.
  /// In terms of performance it is better to not actually do anything.
  /// In terms of profiling/memoization we can exploit the infomration
  /// produced during the false condition to predict approximations
  /// in the future.

  /// The entire runtime is implemented as a dataflow mechanism. The data are
  /// encapuslated in an "instance". This instance is then proccessed on
  /// different phases
  registerInstance("The-Compiler-Needs-To-Create-This-Identifier", accFn,
                   perfoFn, arg, cond, perfo, memo_type, var_info, num_deps);
  /// Phase 1: PreProcessing. This step will preprocess data
  /// to match the data for the approximation technique.
  PreProcess();
  /// Phase 2: Execute the code (approximated or accurately)
  Execute();
  /// Phase 3: Post Process the results (For example stored output computed
  /// values on the runtime)
  PostProcess();

  /// Drop any references to application side memory locations.
  unRegisterInstance();
}
