#ifndef QUEUE_H
#define QUEUE_H

#include "def.h"
#include "data.h"

using namespace std;

class Queue {
  public:
        Queue (void);
        ~Queue (void);
        
        void Insert (Data *in);
        Data* Remove (void);
        Data* Snoop (void);
        
  private:
        void  Add (Data *in);
        Data* Del (void);

	Data* buf[QUEUESIZE];
	long  head;
        long  tail;
	
        int fullFlag;
        int emptyFlag;
	
        pthread_mutex_t *mut;
	
        pthread_cond_t *full;
        pthread_cond_t *empty;
        
};


#endif