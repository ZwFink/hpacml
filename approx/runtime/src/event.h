#ifndef EVENT_H
#define EVENT_H

#include <string>
#include <map>
#include <cuda_runtime.h>
#include <chrono>

namespace EventRecorder {

    class CPUEvent {
    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        std::chrono::time_point<std::chrono::high_resolution_clock> end;
        std::string name;

    public:
        void recordStart();
        void recordEnd();
        float elapsedTime();
        CPUEvent(std::string name);
        CPUEvent();
        std::string getName();
        void setName(std::string name);
    };

    class GPUEvent {
    private:
        cudaEvent_t Start;
        cudaEvent_t End;
        std::string name;

    public:
        GPUEvent();
        GPUEvent(std::string name);
        void recordStart(cudaStream_t stream = 0);
        void recordEnd(cudaStream_t stream = 0);
        float elapsedTime();
        std::string getName();
        void setName(std::string name);
    };

    CPUEvent CreateEvent(const std::string name);
    GPUEvent CreateGPUEvent(const std::string name);
    void LogEvent(GPUEvent e);
    void LogEvent(CPUEvent e);

}
#endif // EVENT_H
