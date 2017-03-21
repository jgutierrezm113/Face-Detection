#ifndef PTHREADS_H
#define PTHREADS_H

#include "include.h"
#include "def.h"
#include "data.h"
#include "queue.h"
#include "auxiliar.h"
#include "../net/bnet.h"
#include "../net/pnet.h"
#include "../net/rnet.h"
#include "../net/onet.h"

/*
 * Pthread functions
 */
 
void* preprocess        (void *ptr);
void* output            (void *ptr);
void* pnet              (void *ptr);
void* rnet              (void *ptr);
void* onet              (void *ptr);

// PNET Struct for queues
typedef struct {
        Queue* queue;
        float scale;
} pnet_info;

// RNET Struct for image resize pthreads 
typedef struct {
        Queue* queue;
        float scale;
} rnet_info;

#endif