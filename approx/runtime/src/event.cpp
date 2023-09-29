#include "event.h"

#include <iostream>
#include <chrono>

namespace EventRecorder {

    void CPUEvent::recordStart() {
        start = std::chrono::high_resolution_clock::now();
    }

    void CPUEvent::recordEnd() {
        end = std::chrono::high_resolution_clock::now();
    }

    float CPUEvent::elapsedTime() {
        // get in microseconds and convert to milliseconds
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }

    std::string CPUEvent::getName() {
        return name;
    }

    void CPUEvent::setName(std::string name) {
        this->name = name;
    }

    CPUEvent::CPUEvent(std::string name) {
        setName(name);
    }

    CPUEvent::CPUEvent() : CPUEvent("Event") {}

    CPUEvent CreateEvent(const std::string name) {
        return CPUEvent(name);
    }

    GPUEvent::GPUEvent() : GPUEvent("Event") {}

    GPUEvent::GPUEvent(std::string name) {
        setName(name);
        cudaEventCreate(&Start);
        cudaEventCreate(&End);
    }

    void GPUEvent::recordStart(cudaStream_t stream) {
        cudaEventRecord(Start, stream);
    }

    void GPUEvent::recordEnd(cudaStream_t stream) {
        cudaEventRecord(End, stream);
    }

    float GPUEvent::elapsedTime() {
        float ms = 0.0;
        cudaEventSynchronize(End);
        cudaEventElapsedTime(&ms, Start, End);
        return ms;
    }

    std::string GPUEvent::getName() {
        return name;
    }

    void GPUEvent::setName(std::string name) {
        this->name = name;
    }

    GPUEvent CreateGPUEvent(const std::string name) {
        GPUEvent e;
        e.setName(name);
        return e;
    }

    void LogEvent(GPUEvent e) {
        std::cout << "EVENT " << e.getName() << ": " << e.elapsedTime() << "ms\n";
    }

    void LogEvent(CPUEvent e) {
        std::cout << "EVENT " << e.getName() << ": " << e.elapsedTime() << "ms\n";
    }
}
