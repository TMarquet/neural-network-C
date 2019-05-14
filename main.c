#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mnist.h"
#include "network.h"

// The main attraction
int main(){

	// Random seed
	srand(time(NULL));

	// Load in training and test data
	printf("Reading data...\n");

	unsigned char* training_labels;
	unsigned char** training_images;
	unsigned char* test_labels;
	unsigned char** test_images;

	loadMNIST(&training_labels, &training_images, &test_labels, &test_images);

	printf("Read complete.\n");

	// Convert data to NN-friendly format
	printf("Converting data...\n");

	double** training_input;
	double** training_output;
	double** test_input;
	double** test_output;

	convertMNIST(&training_labels, &training_images, &test_labels, &test_images,
				&training_input, &training_output, &test_input, &test_output);

	printf("Conversion complete.\n");

	// Prepare network
	int num_layers = 3;
	int sizes[] = {784, 30, 10};

	Network* network = initNetwork(num_layers, sizes);
	//saveNetwork(network, PARAM_NAME);
	//Network* network = loadNetwork(PARAM_NAME);

	// Test network out of the box
	int test_size = TEST_SIZE; // 10_000 max size
	int success = evaluate(network, test_input, test_output, test_size);
	printf("Successfully predicted %d / %d.\n", success, test_size);

	// Train the network
	int training_size = TRAIN_SIZE;	// 60_000 max size
	int mini_batch_size = 10;
	int epochs = 30;
	double learning_rate = 3.0;

	stochasticGradientDescent(network, training_input, training_output, training_size, 
								mini_batch_size, epochs, learning_rate);

	// Test network after training
	success = evaluate(network, test_input, test_output, test_size);
	printf("Successfully predicted %d / %d.\n", success, test_size);

	// Free the memory
	freeMNIST(&training_input, &training_output, &test_input, &test_output);
	free(network);
	
	return 0;
}

/***************************\
*	TO DO LIST
*
* - Implement memset() where setting values to zero
* - Set values to zero before doing matrix multiplications
* - Preallocate memory where malloc and calloc used over and over again
* - Ensure all memory is free() where malloc or calloc used
* - Implement faster matrix multiplicaiton
* - feedForward()
*		- Make the forward propagation look more like that from backPropagation()
*
* - SGD()
*		- Properly shuffle training data to have randomized mini-batches
*		- Impletement optional validation data and end of every epoch
*
* - Overall make code more C like and less Python like
*
\***************************/
