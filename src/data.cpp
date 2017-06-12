#include "../include/data.h"

/*
        GENERAL FUNCTIONS
        *****************
*/
Data::Data(){
  // Initialize synchronization variables
  mut = (pthread_mutex_t *) malloc (sizeof (pthread_mutex_t));
  pthread_mutex_init (mut, NULL);
  counter = 0;
  done = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
  pthread_cond_init (done, NULL);

}

Data::~Data(){
  // Free resources from Packet data
  frame.release();
  processed_frame.release();
  // FIXME: COND destroy is not working properly. Hangs
  //pthread_mutex_destroy (mut);
  //pthread_cond_destroy (done);

}
