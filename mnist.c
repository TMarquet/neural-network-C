#include <stdio.h>
#include <stdlib.h>
#include <byteswap.h>
#include "mnist.h"

void loadMNIST(unsigned char** training_labels, unsigned char*** training_images, 
				unsigned char** test_labels, unsigned char *** test_images){

	// Arrays to store the MNIST data
	training_labels[0] = (unsigned char*)malloc(TRAIN_SIZE * sizeof(unsigned char));
	training_images[0] = (unsigned char**)malloc(TRAIN_SIZE * sizeof(unsigned char*));
	for(int i = 0; i < TRAIN_SIZE; i++)
		training_images[0][i] = (unsigned char*)malloc(784 * sizeof(unsigned char));

	test_labels[0] = (unsigned char*)malloc(TEST_SIZE * sizeof(unsigned char));
	test_images[0] = (unsigned char**)malloc(TEST_SIZE * sizeof(unsigned char*));
	for(int i = 0; i < TEST_SIZE; i++)
		test_images[0][i] = (unsigned char*)malloc(784 * sizeof(unsigned char));

	// Header variables
	unsigned int magic_number, num_images, num_rows, num_cols;
	FILE* fptr;


	// Load training labels
	fptr = fopen(TRAINING_LABEL_NAME, "rb");

	if (!fptr){
		printf("Could not open %s.\n", TRAINING_LABEL_NAME);
		training_labels[0] = NULL;
		training_images[0] = NULL;
		test_labels[0] = NULL;
		test_images[0] = NULL;
		return;
	}

	// Read header values
	fread(&magic_number, 1, sizeof(unsigned int), fptr);
	fread(&num_images, 1, sizeof(unsigned int), fptr);

	// I have an Intel processor R.I.P.
	magic_number = __bswap_32(magic_number);
	num_images = __bswap_32(num_images);

	//printf("%u, %u\n", magic_number, num_images);

	// Read labels
	for(int i = 0; i < num_images; i++)
		fread(training_labels[0] + i, 1, 1, fptr);
	fclose(fptr);


	// Load training images
	fptr = fopen(TRAINING_IMAGE_NAME, "rb");

	if (!fptr){
		printf("Could not open %s.\n", TRAINING_IMAGE_NAME);
		training_labels[0] = NULL;
		training_images[0] = NULL;
		test_labels[0] = NULL;
		test_images[0] = NULL;
		return;
	}

	// Read header values
	fread(&magic_number, 1, sizeof(unsigned int), fptr);
	fread(&num_images, 1, sizeof(unsigned int), fptr);
	fread(&num_rows, 1, sizeof(unsigned int), fptr);
	fread(&num_cols, 1, sizeof(unsigned int), fptr);

	// I have an Intel processor R.I.P.
	magic_number = __bswap_32(magic_number);
	num_images = __bswap_32(num_images);
	num_rows = __bswap_32(num_rows);
	num_cols = __bswap_32(num_cols);

	//printf("%u, %u, %u, %u\n", magic_number, num_images, num_rows, num_cols);

	// Read images
	for(int i = 0; i < num_images; i++)
		for(int j = 0; j < num_rows * num_cols; j++)
				fread(training_images[0][i] + j, 1, 1, fptr);
	fclose(fptr);


	// Load test labels
	fptr = fopen(TEST_LABEL_NAME, "rb");

	if (!fptr){
		printf("Could not open %s.\n", TEST_LABEL_NAME);
		training_labels[0] = NULL;
		training_images[0] = NULL;
		test_labels[0] = NULL;
		test_images[0] = NULL;
		return;
	}

	// Read header values
	fread(&magic_number, 1, sizeof(unsigned int), fptr);
	fread(&num_images, 1, sizeof(unsigned int), fptr);

	// I have an Intel processor R.I.P.
	magic_number = __bswap_32(magic_number);
	num_images = __bswap_32(num_images);

	//printf("%u, %u\n", magic_number, num_images);

	// Read labels
	for(int i = 0; i < num_images; i++)
		fread(test_labels[0] + i, 1, 1, fptr);
	fclose(fptr);


	// Load test images
	fptr = fopen(TEST_IMAGE_NAME, "rb");

	if (!fptr){
		printf("Could not open %s.\n", TEST_IMAGE_NAME);
		training_labels[0] = NULL;
		training_images[0] = NULL;
		test_labels[0] = NULL;
		test_images[0] = NULL;
		return;
	}

	// Read header values
	fread(&magic_number, 1, sizeof(unsigned int), fptr);
	fread(&num_images, 1, sizeof(unsigned int), fptr);
	fread(&num_rows, 1, sizeof(unsigned int), fptr);
	fread(&num_cols, 1, sizeof(unsigned int), fptr);

	// I have an Intel processor R.I.P.
	magic_number = __bswap_32(magic_number);
	num_images = __bswap_32(num_images);
	num_rows = __bswap_32(num_rows);
	num_cols = __bswap_32(num_cols);

	//printf("%u, %u, %u, %u\n", magic_number, num_images, num_rows, num_cols);

	// Read images
	for(int i = 0; i < num_images; i++)
		for(int j = 0; j < num_rows * num_cols; j++)
				fread(test_images[0][i] + j, 1, 1, fptr);
	fclose(fptr);

	return;
}

void convertMNIST(unsigned char** training_labels, unsigned char*** training_images, 
					unsigned char** test_labels, unsigned char*** test_images,
					double*** training_input, double*** training_output, 
					double*** test_input, double*** test_output){

	// Covert chars to doubles
	training_input[0] = (double**)malloc(TRAIN_SIZE * sizeof(double*));
	training_output[0] = (double**)malloc(TRAIN_SIZE * sizeof(double*));
	for(int i = 0; i < TRAIN_SIZE; i++){

		training_input[0][i] = (double*)malloc(784 * sizeof(double));
		for(int j = 0; j < 784; j++)
			training_input[0][i][j] = (double)training_images[0][i][j] / 255.0;

		training_output[0][i] = (double*)calloc(10, sizeof(double));
		training_output[0][i][training_labels[0][i]] = 1;
	}

	test_input[0] = (double**)malloc(TEST_SIZE * sizeof(double*));
	test_output[0] = (double**)malloc(TEST_SIZE * sizeof(double*));
	for(int i = 0; i < TEST_SIZE; i++){

		test_input[0][i] = (double*)malloc(784 * sizeof(double));
		for(int j = 0; j < 784; j++)
			test_input[0][i][j] = (double)test_images[0][i][j] / 255.0;

		test_output[0][i] = (double*)calloc(10, sizeof(double));
		test_output[0][i][test_labels[0][i]] = 1;
	}

	// Free the old char memory
	for(int i = 0; i < TRAIN_SIZE; i++)
		free(training_images[0][i]);
	for(int i = 0; i < TEST_SIZE; i++)
		free(test_images[0][i]);
	free(training_labels[0]);
	free(training_images[0]);
	free(test_labels[0]);
	free(test_images[0]);

	return;
}

void freeMNIST(double*** training_input, double*** training_output, 
					double*** test_input, double*** test_output){

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
