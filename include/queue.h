#ifndef QUEUE_H
#define QUEUE_H

using namespace std;

// Note: Using template for queue. All functions have to be declared in the same
// file.

// Maximum Elements in the Queue
#define QUEUESIZE 10

template <class Element>
class Queue {
	public:
		Queue (void){
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
		~Queue (void){
			pthread_mutex_destroy (mut);
			free (mut);

			pthread_cond_destroy (full);
			free (full);

			pthread_cond_destroy (empty);
			free (empty);
		}
		void Insert (Element in){
			pthread_mutex_lock(mut);       /* protect buffer */

			while (fullFlag == 1 ) /* If there is something
														in the buffer then wait */
				pthread_cond_wait(full, mut);
			Add(in);

			pthread_cond_signal(empty);	/* wake up consumer */
			pthread_mutex_unlock(mut);	/* release the buffer */
		}
		Element Remove (void){
			pthread_mutex_lock(mut);
			while (emptyFlag == 1)
				pthread_cond_wait(empty, mut);
			Element out = Del();
			pthread_cond_signal(full);
			pthread_mutex_unlock(mut);
			return (out);
		}
		Element Snoop (void){
			pthread_mutex_lock(mut);
			while (emptyFlag == 1)
				pthread_cond_wait(empty, mut);
			Element out = buf[head];
			pthread_cond_signal(full);
			pthread_mutex_unlock(mut);
			return (out);
		}

	private:
		void  Add (Element in){
			buf[tail] = in;

			tail++;
			if (tail == QUEUESIZE)
				tail = 0;

			if (tail == head){
				fullFlag = 1;
			}
			emptyFlag = 0;

			return;
		}
		Element Del (void){
			Element out = buf[head];

			head++;
			if (head == QUEUESIZE)
				head = 0;

			if (head == tail)
				emptyFlag = 1;
			fullFlag = 0;

			return out;
		}

		Element buf[QUEUESIZE];
		long  head;
		long  tail;

		int fullFlag;
		int emptyFlag;

		pthread_mutex_t *mut;

		pthread_cond_t *full;
		pthread_cond_t *empty;

};

#endif
