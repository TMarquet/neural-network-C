#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <byteswap.h>
#include "mnist.h"

void loadMNIST(double*** training_input, double*** training_output, double*** test_input, double*** test_output){

	// Char arrays to store the MNIST data
	unsigned char* training_labels = (unsigned char*)malloc(TRAIN_SIZE * sizeof(unsigned char));
	unsigned char** training_images = (unsigned char**)malloc(TRAIN_SIZE * sizeof(unsigned char*));
	for(int i = 0; i < TRAIN_SIZE; i++)
		training_images[i] = (unsigned char*)malloc(784 * sizeof(unsigned char));

	unsigned char* test_labels = (unsigned char*)malloc(TEST_SIZE * sizeof(unsigned char));
	unsigned char** test_images = (unsigned char**)malloc(TEST_SIZE * sizeof(unsigned char*));
	for(int i = 0; i < TEST_SIZE; i++)
		test_images[i] = (unsigned char*)malloc(784 * sizeof(unsigned char));

	// Header variables
	unsigned int magic_number, num_images, num_rows, num_cols;
	FILE* fptr;


	// Load training labels
	fptr = fopen(TRAINING_LABEL_NAME, "rb");

	if (!fptr){
		printf("Could not open %s.\n", TRAINING_LABEL_NAME);
		training_labels = NULL;
		training_images = NULL;
		test_labels = NULL;
		test_images = NULL;
		return;
	}

	// Read header values
	fread(&magic_number, 1, sizeof(unsigned int), fptr);
	fread(&num_images, 1, sizeof(unsigned int), fptr);

	// I have an Intel processor R.I.P.
	num_images = __bswap_32(num_images);

	// Read labels
	fread(training_labels, 1, num_images, fptr);
	fclose(fptr);


	// Load training images
	fptr = fopen(TRAINING_IMAGE_NAME, "rb");

	if (!fptr){
		printf("Could not open %s.\n", TRAINING_IMAGE_NAME);
		training_labels = NULL;
		training_images = NULL;
		test_labels = NULL;
		test_images = NULL;
		return;
	}

	// Read header values
	fread(&magic_number, 1, sizeof(unsigned int), fptr);
	fread(&num_images, 1, sizeof(unsigned int), fptr);
	fread(&num_rows, 1, sizeof(unsigned int), fptr);
	fread(&num_cols, 1, sizeof(unsigned int), fptr);

	// I have an Intel processor R.I.P.
	num_images = __bswap_32(num_images);
	num_rows = __bswap_32(num_rows);
	num_cols = __bswap_32(num_cols);

	// Read images
	for(int i = 0; i < num_images; i++)
		fread(training_images[i], 1, num_rows*num_cols, fptr);
	fclose(fptr);


	// Load test labels
	fptr = fopen(TEST_LABEL_NAME, "rb");

	if (!fptr){
		printf("Could not open %s.\n", TEST_LABEL_NAME);
		training_labels = NULL;
		training_images = NULL;
		test_labels = NULL;
		test_images = NULL;
		return;
	}

	// Read header values
	fread(&magic_number, 1, sizeof(unsigned int), fptr);
	fread(&num_images, 1, sizeof(unsigned int), fptr);

	// I have an Intel processor R.I.P.
	num_images = __bswap_32(num_images);

	// Read labels
	fread(test_labels, 1, num_images, fptr);
	fclose(fptr);


	// Load test images
	fptr = fopen(TEST_IMAGE_NAME, "rb");

	if (!fptr){
		printf("Could not open %s.\n", TEST_IMAGE_NAME);
		training_labels = NULL;
		training_images = NULL;
		test_labels = NULL;
		test_images = NULL;
		return;
	}

	// Read header values
	fread(&magic_number, 1, sizeof(unsigned int), fptr);
	fread(&num_images, 1, sizeof(unsigned int), fptr);
	fread(&num_rows, 1, sizeof(unsigned int), fptr);
	fread(&num_cols, 1, sizeof(unsigned int), fptr);

	// I have an Intel processor R.I.P.
	num_images = __bswap_32(num_images);
	num_rows = __bswap_32(num_rows);
	num_cols = __bswap_32(num_cols);

	// Read images
	for(int i = 0; i < num_images; i++)
		fread(test_images[i], 1, num_rows*num_cols, fptr);
	fclose(fptr);


	// Convert char data to doubles
	training_input[0] = (double**)malloc(TRAIN_SIZE * sizeof(double*));
	training_output[0] = (double**)malloc(TRAIN_SIZE * sizeof(double*));
	for(int i = 0; i < TRAIN_SIZE; i++){

		training_input[0][i] = (double*)malloc(784 * sizeof(double));
		for(int j = 0; j < 784; j++)
			training_input[0][i][j] = (double)training_images[i][j] / 255.0;

		training_output[0][i] = (double*)calloc(10, sizeof(double));
		training_output[0][i][training_labels[i]] = 1;
	}

	test_input[0] = (double**)malloc(TEST_SIZE * sizeof(double*));
	test_output[0] = (double**)malloc(TEST_SIZE * sizeof(double*));
	for(int i = 0; i < TEST_SIZE; i++){

		test_input[0][i] = (double*)malloc(784 * sizeof(double));
		for(int j = 0; j < 784; j++)
			test_input[0][i][j] = (double)test_images[i][j] / 255.0;

		test_output[0][i] = (double*)calloc(10, sizeof(double));
		test_output[0][i][test_labels[i]] = 1;
	}

	// Free the char memory
	for(int i = 0; i < TRAIN_SIZE; i++)
		free(training_images[i]);
	for(int i = 0; i < TEST_SIZE; i++)
		free(test_images[i]);
	free(training_labels);
	free(training_images);
	free(test_labels);
	free(test_images);

	return;
}

void freeMNIST(double*** training_input, double*** training_output, double*** test_input, double*** test_output){

	for(int i = 0; i < TRAIN_SIZE; i++){
		free(training_input[0][i]);
		free(training_output[0][i]);
	}
	for(int i = 0; i < TEST_SIZE; i++){
		free(test_input[0][i]);
		free(test_output[0][i]);
	}
	free(training_input[0]);
	free(training_output[0]);
	free(test_input[0]);
	free(test_output[0]);

	return;
}

void showMNIST(double** training_input, double** training_output, int num_images){

	int image;
	double threshhold = 0.5;

	for(int i = 0; i < num_images; i++){

		image = rand() % 5000;

		for(int i = 0; i < 28; i++){
			for(int j = 0; j < 28; j++){

				if (training_input[image][28*i + j] > threshhold)
					printf("# ");
				else
					printf("  ");
			}
			printf("\n");
		}

		printf("This image corresponds to: ");
		for(int i = 0; i < 10; i++)
			if (training_output[image][i] > 0.5){
				printf("%d\n", i);
				break;
			}
	}
	return;
}

void benchmarkLoadMNIST(double*** training_input, double*** training_output, double*** test_input, double*** test_output){

	clock_t t = clock(); 
	loadMNIST(training_input, training_output, test_input, test_output);
	t = clock() - t; 
	double time_taken = ((double)t)/CLOCKS_PER_SEC;
	
	printf("loadMNIST() took %lf seconds to execute \n", time_taken); 
	return;
}
