#ifndef __APPROX_DATA__
#define __APPROX_DATA__

#include <chrono>
#include <iostream>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <approx_types.h>

class ApproximateTechniqueBase;

enum MType {
// OpenCL image types
#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix) Id,
#include "clang/Basic/OpenCLImageTypes.def"
// OpenCL extension types
#define EXT_OPAQUE_TYPE(ExtType, Id, Ext) Id,
#include "clang/Basic/OpenCLExtensionTypes.def"
// SVE Types
#define SVE_TYPE(Name, Id, SingletonId) Id,
#include "clang/Basic/AArch64SVEACLETypes.def"
// All other builtin types
#define BUILTIN_TYPE(Id, SingletonId) Id,
#define LAST_BUILTIN_TYPE(Id) LastKind = Id
#include "clang/AST/BuiltinTypes.def"
};

enum Directionality : int { Input = 1, Output = 2, InputOutput = 4 };

class TypeInterface {
protected:
  /// The pointer pointing to the application data memory.
  /// The pointer is consider as "live" in the following scnenario:
  /// input pointers -> live when entering the region
  /// output pointers -> live on exit of the region
  /// in/out pointers -> live on input and output of each region.
  /// Finally in/out pointers will never exist in the runtime level
  /// a pointer can be either in or out. What it changes is the liveness
  /// of the pointer.
  void *origPtr;
  /// The dPtr is a pointer pointing to a copied version of the data. This
  /// is used to get information of previous accessed data. For example in out
  /// memoization I need to have access on previous output data.
  void *dPtr;
  /// The type of the pointer flaot/double/int etc.
  MType dType;
  /// The number of elements
  size_t nElements;
  /// The size of the element
  size_t sElement;
  /// The size of the memory region = sElement * nElements;
  size_t sRegion;
  /// Is this an input output or input/output data element;
  Directionality dDirection;
  int id;

public:
  TypeInterface(void *ptr, MType dataType, size_t nElements, size_t sElement,
                size_t sRegion, int id)
      : origPtr(ptr), dPtr(nullptr), dType(dataType), nElements(nElements),
        sElement(sElement), sRegion(sRegion), id(id) {
    assert(sElement * nElements == sRegion);
  }

  virtual ~TypeInterface() {
    dPtr = nullptr;
    origPtr = nullptr;
  }

  virtual void allocatePtr() = 0;

  void setDirection(Directionality dir) { dDirection = dir; }

  void describe() {
    std::cout << "ID:" << id << std::endl;
    std::cout << "sRegion :" << sRegion << std::endl;
    std::cout << "nElements :" << nElements << std::endl;
    std::cout << "sElement :" << sElement << std::endl;
    std::cout << "DataType :" << dType << std::endl;
  }
  virtual void copy() = 0;

  virtual void dump() = 0;

protected:
  void setOpaquePtr(void *ptr) {
    assert(ptr != nullptr);
    dPtr = ptr;
  }
};

template <class T> class Type : public TypeInterface {
  T *dtPtr;

public:
  Type(void *ptr, MType dataType, size_t nElements, size_t sElement,
       size_t sRegion, int id)
      : TypeInterface(ptr, dataType, nElements, sElement, sRegion, id), dtPtr(nullptr) {
    assert(sElement == sizeof(T));
  }

  ~Type() {
    if (dtPtr != nullptr) {
      delete[] dtPtr;
      dtPtr = nullptr;
    }
  }

  virtual void allocatePtr() {
    dtPtr = new T[nElements];
    setOpaquePtr((void *)dtPtr);
  }

  void copy() {
    assert(dPtr);
    assert(origPtr);
    if (nElements == 1) {
      dtPtr[0] = *((T *)origPtr);
    } else {
      memcpy(dtPtr, origPtr, sRegion);
    }
  }

  void dump() {
    for (size_t i = 0; i < nElements; i++) {
      std::cout << ":" << dtPtr[i];
    }
  }
};

/// ApproxInstance encapsulates a single call
/// to an approximation region.
class ApproxInstance {
  /// All data values passed as in/out/inout
  std::vector<TypeInterface *> allValues;
  std::vector<TypeInterface *> inputValues;
  std::vector<TypeInterface *> outputValues;
  /// Name of the data region.
  /// This is created by the compiler
  std::string name;
  /// This is an identifier for the instantiation
  /// of this approximation region. It will be usefull
  /// to keep track of the history of each region.
  uint64_t count;

  std::chrono::time_point<std::chrono::high_resolution_clock> startT;
  std::chrono::time_point<std::chrono::high_resolution_clock> endT;
  std::chrono::duration<double> elapsed;

public:
  void (*accurateFn)(void *);
  void (*perfoFn)(void *);
  approx_perfo_info_t *perfoArgs;
  void *argsFn;
  bool cond;
  /// TODO this needs to be changed and we should
  /// use the types included by the compiler. To make
  /// sure that we are in sync.
  int memo_type;

  bool isApproximated;


public:
  using iterator = std::vector<TypeInterface *>::iterator;
  using const_iterator = std::vector<TypeInterface *>::const_iterator;

  ApproxInstance(std::string name, uint64_t count, void (*accFn)(void *),
                 void (*perfoFn)(void *), void *arg, bool cond,
                 approx_perfo_info_t *perfoArgs, int memo_type)
      : name(name), count(count), accurateFn(accFn), perfoFn(perfoFn),
        perfoArgs(perfoArgs), argsFn(arg), memo_type(memo_type),
        isApproximated(false) {}
  ~ApproxInstance() {
    for (TypeInterface *V : allValues) {
      delete V;
    }
    allValues.clear();
    inputValues.clear();
    outputValues.clear();
    accurateFn = nullptr;
    perfoFn = nullptr;
    argsFn = nullptr;
  }

  void addValue(TypeInterface *value, Directionality direction) {
    allValues.push_back(value);
    if (direction == Input) {
      inputValues.push_back(value);
      value->setDirection(direction);
    } else if (direction == Output) {
      outputValues.push_back(value);
      value->setDirection(direction);
    }
  }

  iterator begin() { return allValues.begin(); }
  iterator end() { return allValues.end(); }

  iterator ibegin() { return inputValues.begin(); }
  iterator iend() { return inputValues.end(); }

  iterator obegin() { return outputValues.begin(); }
  iterator oend() { return outputValues.end(); }

  const_iterator cbegin() { return allValues.cbegin(); }
  const_iterator cend() { return allValues.cend(); }

  const_iterator icbegin() { return inputValues.cbegin(); }
  const_iterator icend() { return inputValues.cend(); }

  const_iterator ocbegin() { return outputValues.cbegin(); }
  const_iterator ocend() { return outputValues.cend(); }

  void debug_info() {
    std::cout << "================ All Data ===================" << std::endl;
    for (ApproxInstance::iterator it = begin(); it != end(); it++) {
      (*it)->describe();
      std::cout << std::endl;
    }

    std::cout << "================ Input Data ===================" << std::endl;
    for (ApproxInstance::iterator it = ibegin(); it != iend(); it++) {
      (*it)->describe();
      std::cout << std::endl;
    }

    std::cout << "================ Output Data ==================="
              << std::endl;
    for (ApproxInstance::iterator it = obegin(); it != oend(); it++) {
      (*it)->describe();
      std::cout << std::endl;
    }
  }

  void registerInputs() {
    for (ApproxInstance::iterator it = ibegin(); it != iend(); it++) {
      (*it)->allocatePtr();
      (*it)->copy();
    }
  }

  void registerOutputs() {
    for (ApproxInstance::iterator it = obegin(); it != oend(); it++) {
      (*it)->allocatePtr();
      (*it)->copy();
    }
  }

  void setApproximated(bool val) { isApproximated = val; }
  bool getApproximated() { return isApproximated; }

  void startTime() { startT = std::chrono::high_resolution_clock::now(); }

  void endTime() {
    endT = std::chrono::high_resolution_clock::now();
    elapsed = endT - startT;
  }

  void dump() {
    std::cout << count;
    for (ApproxInstance::iterator it = ibegin(); it != iend(); it++) {
      (*it)->dump();
    }

    for (ApproxInstance::iterator it = obegin(); it != oend(); it++) {
      (*it)->dump();
    }
    std::cout << "\n";
  }

  void unRegisterInstance() {
    accurateFn = nullptr;
    perfoFn = nullptr;
    argsFn = nullptr;
  }

  std::string getName() { return name; }
};

#endif