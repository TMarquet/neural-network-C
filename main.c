#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mnist.h"
#include "network.h"

// The main attraction
int main(){

	// Random seed
	srand(time(NULL));

	// Load MNIST data
	printf("Loading MNIST data...\n");
	double **training_input, **training_output, **test_input, **test_output;
	loadMNIST(&training_input, &training_output, &test_input, &test_output);

	// Show that the MNIST load worked
	showMNIST(training_input, training_output, 0);

	// Prepare network
	int num_layers = 3;
	int sizes[] = {784, 30, 10};

	Network* network = initNetwork(num_layers, sizes);
	//saveNetwork(network, PARAM_NAME);
	//Network* network = loadNetwork(PARAM_NAME);

	// Test network out of the box
	int test_size = TEST_SIZE;		// 10_000 max size
	evaluate(network, test_input, test_output, test_size);

	// Train the network
	int training_size = 10000;		// 60_000 max size
	int mini_batch_size = 10;
	int epochs = 1;
	double learning_rate = 3.0;

	// Time the training process
	benchmarkSGD(network, training_input, training_output, training_size, 
								mini_batch_size, epochs, learning_rate);

	// Test network after training
	evaluate(network, test_input, test_output, test_size);

	// Free training memory
	freeMNIST(&training_input, &training_output, &test_input, &test_output);

	return 0;
}

/***************************\
*	TO DO LIST
*
* - Implement faster matrix multiplicaiton
*
* - SGD()
*		- Properly shuffle training data to have randomized mini-batches
*		- Impletement optional validation data and end of every epoch
*
* - Overall make code more C like and less Python like
*
\***************************/
