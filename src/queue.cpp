
#include "../include/queue.h"

Queue::Queue (void){

	emptyFlag = 1;
	fullFlag = 0;
	head = 0;
	tail = 0;
        
	mut = (pthread_mutex_t *) malloc (sizeof (pthread_mutex_t));
	pthread_mutex_init (mut, NULL);
        
	full = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
	pthread_cond_init (full, NULL);
        
	empty = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
	pthread_cond_init (empty, NULL);
}

Queue::~Queue (void){

	pthread_mutex_destroy (mut);
	free (mut);	
        
	pthread_cond_destroy (full);
	free (full);
        
	pthread_cond_destroy (empty);
	free (empty);
}

void Queue::Insert (Data *in){
        pthread_mutex_lock(mut);       /* protect buffer */
        
        while (fullFlag == 1 ) /* If there is something 
                              in the buffer then wait */
                pthread_cond_wait(full, mut);
        Add(in);
        
        pthread_cond_signal(empty);	/* wake up consumer */
        pthread_mutex_unlock(mut);	/* release the buffer */
}
        
Data* Queue::Remove (void){
        pthread_mutex_lock(mut);
        while (emptyFlag == 1)
                pthread_cond_wait(empty, mut);
        Data* out = Del();
        pthread_cond_signal(full);
        pthread_mutex_unlock(mut);
        return (out);
}

void Queue::Add (Data *in) {
	
        buf[tail] = in;
        
	tail++;
	if (tail == QUEUESIZE)
		tail = 0;
        
	if (tail == head){
		fullFlag = 1;
                #if(DEBUG_ENABLED)
                        //FIXME: Fatal error, we should never have an over flow
                        //cout << "JUMP!! SOMETHING WENT WRONG!\n";
                #endif
        }
	emptyFlag = 0;

	return;
}

Data* Queue::Del (void) {
	Data *out = buf[head];

	head++;
	if (head == QUEUESIZE)
		head = 0;
        
	if (head == tail)
		emptyFlag = 1;
	fullFlag = 0;

	return out;
}

Data* Queue::Snoop (void){
        pthread_mutex_lock(mut);
        while (emptyFlag == 1)
                pthread_cond_wait(empty, mut);
	Data *out = buf[head];
        pthread_cond_signal(full);
        pthread_mutex_unlock(mut);
        return (out);
}