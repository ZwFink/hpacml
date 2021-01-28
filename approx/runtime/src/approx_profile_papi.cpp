#include <approx_profile.h>
#include <cstdlib>
#include <cstring>
#include <hdf5.h>
#include <iostream>
#include <papi.h>
#include <string>
#include <limits.h>

#include <approx_profile_papi.h>

#include "H5Gpublic.h"
#include "io_helpers.h"

#define PAPI_MAX_STRING_NAME 200
#define DELIM ";"

PapiProfiler::PapiProfiler(const char *FName) {
    FileId = openHDF5File(FName);
    ProfileEvents = PAPI_NULL;
    int ErrorCode = PAPI_library_init(PAPI_VER_CURRENT);
    if (ErrorCode != PAPI_VER_CURRENT) {
        fprintf(stderr, "Could not Initialize Library\n %s \n Exiting...\n",
                PAPI_strerror(ErrorCode));
        exit(-1);
    }

    /* Currently nested profiling is not supported.           */
    /* Profiling will start and end when the code enters and  */
    /* exits a  code region respectively. So there is no need */
    /* for an event set for each code region                  */
    if ((ErrorCode = PAPI_create_eventset(&ProfileEvents)) != PAPI_OK) {
        fprintf(stderr, "Could not create Event Set\n %s \n Exiting...\n",
                PAPI_strerror(ErrorCode));
        exit(-1);
    }

    HCounters = PAPI_num_counters();
    const char *EPAPI = std::getenv("PROFILE_EVENTS");

    if ( EPAPI == nullptr ){
        fprintf(stderr, "You need to specify which events I should monitor\nExiting...\n");
        exit(-1);
    }

    EventNames = new char *[HCounters];
    PapiNames = new char[strlen(EPAPI) + 1];
    memcpy(PapiNames, EPAPI, strlen(EPAPI) + 1);
    int i = 0;

    EventNames[i] = std::strtok(PapiNames, DELIM);
    printf("Event name is %s\n", EventNames[i]);
    LEvents = 0;

    while (EventNames[i] != nullptr) {
        EventNames[++i] = std::strtok(NULL, DELIM);
        printf("Event name is %s\n", EventNames[i]);
    }

    LEvents = i;
    if (LEvents > HCounters) {
        fprintf(stderr, "Multiplexing a.t.m is not supported\n");
        exit(-1);
    }

    printf("Events are %d\n", LEvents);

    int Events[LEvents];
    for (i = 0; i < LEvents; i++) {
        if (PAPI_event_name_to_code(EventNames[i], &Events[i]) != PAPI_OK) {
            fprintf(stderr, "Coulot not get event name\n exiting...\n");
            exit(-1);
        }
    }

    if ((ErrorCode = PAPI_add_events(ProfileEvents, Events, LEvents) ) != PAPI_OK) {
        fprintf(stderr, "Could not add events\n%s\n Exiting\n",
                PAPI_strerror(ErrorCode));
        exit(-1);
    }

    TStats = new long long[LEvents];
}

PapiProfiler::~PapiProfiler() {
    // I need to write all the data to the respective
    // group in the HDF5 file.
    for (auto region : AddrToStats) {
        char *RName = region.second->getName();
        double *stats = region.second->getStats();
        unsigned int NInvocations = region.second->getInvocations();
        hid_t GId = createOrOpenGroup(RName, FileId);
        hid_t GProfile = createOrOpenGroup("ProfileData", GId);

        for (int i = 0; i < LEvents; i++) {
            writeProfileData(EventNames[i], GProfile,
                    &stats[i], ESTAT);
        }

        H5Gclose(GId);
        H5Gclose(GProfile);
    }
    H5Fclose(FileId);
}

void PapiProfiler::startProfile(const char *RName, uintptr_t FnAddr) {
    if (PAPI_start(ProfileEvents) != PAPI_OK) {
        fprintf(stderr, "Could not start counters\nExiting...\n");
        exit(-1);
    }
    return;
}

void PapiProfiler::stopProfile(const char *RName, uintptr_t FnAddr) {
    if (PAPI_stop(ProfileEvents, TStats) != PAPI_OK) {
        fprintf(stderr, "Could not stop counters\nExiting...\n");
        exit(-1);
    }

    if (AddrToStats.find(FnAddr) == AddrToStats.end()) {
        RegionProfiler *NRegion = new RegionProfiler(RName, LEvents);
        AddrToStats[FnAddr] = NRegion;
    }
    AddrToStats[FnAddr]->increaseStats(TStats);
    return;
}

RegionProfiler::RegionProfiler(const char *Name, int NStats) : NStats(NStats), Stats(nullptr) {
    RName = new char[strlen(Name) + 1];
    std::strcpy(RName, Name);
}

RegionProfiler::~RegionProfiler() {
    for (auto iter : Accum)
        delete [] iter;

    delete[] Stats;
    delete[] RName;
}

void RegionProfiler::increaseStats(long long *CStats) {
    long long *NData = new long long[NStats];
    std::memcpy(NData, CStats, sizeof(long long)* NStats);
    Accum.push_back(NData);
    return;
}

double *RegionProfiler::getStats(){
    Stats = new double[NStats * ESTAT];

    for (int i = 0; i < NStats; i++){
        Stats[i*ESTAT + AVGI] = 0.0;
        Stats[i*ESTAT + STDI] = 0.0;
        Stats[i*ESTAT + MINI] = std::numeric_limits<double>::max();
        Stats[i*ESTAT + MAXI] = std::numeric_limits<double>::min();
    }

    for (auto iter : Accum ){
        for ( unsigned int i = 0; i < NStats; i++){
            Stats[i*ESTAT + AVGI] += (double) iter[i]/(double) Accum.size();
            if (iter[i] > Stats[i*ESTAT + MAXI]){
                Stats[i*ESTAT + MAXI] = iter[i];
            }

            if ( iter[i]< Stats[i*ESTAT + MINI]){
                Stats[i*ESTAT+MINI] = iter[i];
            }
        }
    }

    for ( auto iter : Accum ){
        for (unsigned int i =0; i < NStats; i++){
            double tmp;
            tmp = (iter[i] - Stats[i*ESTAT + AVGI]);
            tmp *= tmp;
            tmp /= (double) Accum.size();
            Stats[i*ESTAT+STDI] += tmp;
        }
    }

    for (unsigned int i = 0; i < NStats; i++){
        Stats[i*ESTAT+STDI] = sqrt(Stats[i*ESTAT+STDI]);
    }

    return Stats;
}