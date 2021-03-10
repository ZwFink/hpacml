#ifndef __THREAD_STORAGE__
#define __THREAD_STORAGE__

#include <iostream>
using namespace std;

#define MAX_REGIONS 20

template <typename T>
class MemoPool{
  T ***memoRegions;
  int *lastMemoIn;
  int totalThreads;
public:
  MemoPool(){
     totalThreads = omp_get_num_procs(); 
     lastMemoIn = new int[totalThreads]();
     memoRegions = new T**[totalThreads];
     for ( int i = 0; i < totalThreads; i++){
       memoRegions[i] = new T*[MAX_REGIONS];
     }
  };

  ~MemoPool() {
    for (int i = 0; i < totalThreads; i++) {
      for (int j = 0; j < lastMemoIn[i]; j++) {
        delete memoRegions[i][j];
      }
      delete [] memoRegions[i];
    }
    delete [] memoRegions;
    delete [] lastMemoIn;
  }

  T *findMemo(int threadId, unsigned long Addr) {
    static thread_local int myIndex = -1;
    static thread_local T**thread_region = memoRegions[threadId];

    if (myIndex != -1 && thread_region[myIndex] && (((unsigned long) (thread_region[myIndex]->accurate)) == Addr)){
      return thread_region[myIndex];
    }

    for (int i = 0; i < lastMemoIn[threadId]; i++) {
      if ((unsigned long)(thread_region[i]->accurate) == Addr)
        return memoRegions[threadId][i];
    }
    return nullptr;
  }

  T *addNew(int threadId, T* newRegion){
    if ( lastMemoIn[threadId] >= MAX_REGIONS ){
      cout << " I Reach Maximum Regions Exiting..\n";
      exit(-1);
    }
    memoRegions[threadId][lastMemoIn[threadId]++] = newRegion;
    return newRegion;
  }
};
#endif