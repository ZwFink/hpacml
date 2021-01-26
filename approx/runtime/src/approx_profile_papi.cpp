#include <approx_profile.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <string>
#include <papi.h>
#include <hdf5.h>

#include <approx_profile_papi.h>

#include "io_helpers.h"


#define PAPI_MAX_STRING_NAME 200
#define DELIM ";"

PapiProfiler::PapiProfiler(const char *FName){
    FileId = openHDF5File(FName);
    ProfileEvents = PAPI_NULL;
    int ErrorCode = PAPI_library_init(PAPI_VER_CURRENT);
    if ( ErrorCode == PAPI_VER_CURRENT ){
        fprintf(stderr, "Could Not Initialize PAPI Library");
        exit(-1);
    }

    /* Currently nested profiling is not supported.           */
    /* Profiling will start and end when the code enters and  */
    /* exits a  code region respectively. So there is no need */
    /* for an event set for each code region                  */
    if (PAPI_create_eventset(&ProfileEvents) != PAPI_OK){
        fprintf(stderr, "Could not create Event Set");
        exit(-1);
    }

    HCounters = PAPI_num_counters();
    const char *EPAPI = std::getenv("PROFILE_EVENTS");
    BIndex = std::atoi(std::getenv("PAPI_COLUMN"));
    TEvents = std::atoi(std::getenv("PAPI_TOTAL_EVENTS"));
    char *EventNames[HCounters];
    PapiNames = new char[strlen(EPAPI)+1];
    memcpy(PapiNames, EPAPI, strlen(EPAPI)+1);
    int i = 0;

    EventNames[i] = std::strtok(PapiNames, DELIM);
    LEvents = 0;

    while(EventNames[i] != nullptr){
        EventNames[++i] = std::strtok(NULL, DELIM);
    }

    LEvents = i;
    if (LEvents > HCounters) {
        fprintf(stderr, "Multiplexing a.t.m is not supported\n");
        exit(-1);
    }

    int Events[LEvents];
    for ( i = 0; i < LEvents; i++){
        if ( PAPI_event_name_to_code(EventNames[i], &Events[i]) != PAPI_OK) {
            fprintf(stderr, "Coulot not get event name\n exiting...\n");
            exit(-1);
        }
    }

    if ( PAPI_create_eventset(&ProfileEvents) != PAPI_OK) {
        fprintf(stderr, "Could not create event set\n Exiting...\n");
        exit(-1);
    }

    if ( PAPI_add_events(ProfileEvents, Events, LEvents) != PAPI_OK){
        fprintf(stderr, "Could not add events\n Exiting\n");
        exit(-1);
    }

    TStats = new long long [LEvents];
}

PapiProfiler::~PapiProfiler(){
}

void PapiProfiler::startProfile(const char *RName, uintptr_t FnAddr){
    if ( PAPI_start(ProfileEvents) != PAPI_OK){
        fprintf(stderr, "Could not start counters\nExiting...\n");
        exit(-1);
    }
    return;
}

void PapiProfiler::stopProfile(const char *RName, uintptr_t FnAddr){
    if ( PAPI_stop(ProfileEvents, TStats) != PAPI_OK ){
        fprintf(stderr, "Could not stop counters\nExiting...\n");
        exit(-1);
    }

    if (AddrToStats.find(FnAddr) == AddrToStats.end()){
        RegionProfiler *NRegion = new RegionProfiler(RName, LEvents);
        AddrToStats[FnAddr] = NRegion;
    }
    AddrToStats[FnAddr]->increaseStats(TStats);
    return;
}

RegionProfiler::RegionProfiler(const char *Name, int NStats) : NStats(NStats) {
    RName = new char[strlen(Name)+1];
    std::strcpy(RName, Name);
    Accum = new long long [NStats];
    NInvocations = 0;
    memset(Accum, 0, NStats * sizeof(unsigned long long));
}

RegionProfiler::~RegionProfiler(){
    delete [] Accum;
    delete [] RName;
};

void RegionProfiler::increaseStats(long long *CStats){
    for (unsigned int i = 0; i < NStats; i++){
        Accum[i] += CStats[i];
    }
}