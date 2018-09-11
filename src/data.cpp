#include "../include/data.h"

/*
				GENERAL FUNCTIONS
				*****************
*/
Data::Data(){
	// Initialize synchronization variables
	//mut = (pthread_mutex_t *) malloc (sizeof (pthread_mutex_t));
	pthread_mutex_init (&mut, NULL);
	counter = 0;
	//done = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
	pthread_cond_init (&done, NULL);

}

Data::~Data(){
	// Free resources from Packet data
	frame.release();
	processed_frame.release();
	// FIXME: COND destroy is not working properly. Hangs
	pthread_mutex_destroy (&mut);
	pthread_cond_destroy (&done);
	//free(mut);
	//free(done);

}

void Data::WaitForCounter(int num){
	// Wait for Counter to reach a certain number
	pthread_mutex_lock(&mut);
	while (counter < num )
		pthread_cond_wait(&done, &mut);
	pthread_mutex_unlock(&mut);
}

void Data::IncreaseCounter(void){
	// Increase Counter
	pthread_mutex_lock(&mut);
	counter++;
	pthread_cond_signal(&done);
	pthread_mutex_unlock(&mut);
}

void Data::ResetCounter(void){
	// Increase Counter
	pthread_mutex_lock(&mut);
	counter = 0;
	pthread_cond_signal(&done);
	pthread_mutex_unlock(&mut);
}
