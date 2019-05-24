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
		training_images[i] = (unsigned char*)malloc(IMAGE_SIZE * sizeof(unsigned char));

	unsigned char* test_labels = (unsigned char*)malloc(TEST_SIZE * sizeof(unsigned char));
	unsigned char** test_images = (unsigned char**)malloc(TEST_SIZE * sizeof(unsigned char*));
	for(int i = 0; i < TEST_SIZE; i++)
		test_images[i] = (unsigned char*)malloc(IMAGE_SIZE * sizeof(unsigned char));


	// Load training labels
	FILE* fptr = fopen(TRAINING_LABEL_NAME, "rb");

	if (!fptr){
		printf("Could not open %s.\n", TRAINING_LABEL_NAME);
		(*training_input) = NULL;
		(*training_output) = NULL;
		(*test_input) = NULL;
		(*test_output) = NULL;
		return;
	}

	// Skip header values
	fseek(fptr, 8, SEEK_SET);

	// Read labels
	fread(training_labels, 1, TRAIN_SIZE, fptr);
	fclose(fptr);


	// Load training images
	fptr = fopen(TRAINING_IMAGE_NAME, "rb");

	if (!fptr){
		printf("Could not open %s.\n", TRAINING_IMAGE_NAME);
		(*training_input) = NULL;
		(*training_output) = NULL;
		(*test_input) = NULL;
		(*test_output) = NULL;
		return;
	}

	// Skip header values
	fseek(fptr, 16, SEEK_SET);

	// Read images
	for(int i = 0; i < TRAIN_SIZE; i++)
		fread(training_images[i], 1, IMAGE_SIZE, fptr);
	fclose(fptr);


	// Load test labels
	fptr = fopen(TEST_LABEL_NAME, "rb");

	if (!fptr){
		printf("Could not open %s.\n", TEST_LABEL_NAME);
		(*training_input) = NULL;
		(*training_output) = NULL;
		(*test_input) = NULL;
		(*test_output) = NULL;
		return;
	}

	// Skip header values
	fseek(fptr, 8, SEEK_SET);

	// Read labels
	fread(test_labels, 1, TEST_SIZE, fptr);
	fclose(fptr);


	// Load test images
	fptr = fopen(TEST_IMAGE_NAME, "rb");

	if (!fptr){
		printf("Could not open %s.\n", TEST_IMAGE_NAME);
		(*training_input) = NULL;
		(*training_output) = NULL;
		(*test_input) = NULL;
		(*test_output) = NULL;
		return;
	}

	// Skip header values
	fseek(fptr, 16, SEEK_SET);

	// Read images
	for(int i = 0; i < TEST_SIZE; i++)
		fread(test_images[i], 1, IMAGE_SIZE, fptr);
	fclose(fptr);


	// Convert char data to doubles
	(*training_input) = (double**)malloc(TRAIN_SIZE * sizeof(double*));
	(*training_output) = (double**)malloc(TRAIN_SIZE * sizeof(double*));
	for(int i = 0; i < TRAIN_SIZE; i++){

		(*training_input)[i] = (double*)malloc(IMAGE_SIZE * sizeof(double));
		for(int j = 0; j < IMAGE_SIZE; j++)
			(*training_input)[i][j] = (double)training_images[i][j] / 255.0;

		(*training_output)[i] = (double*)calloc(OUTPUT_SIZE, sizeof(double));
		(*training_output)[i][training_labels[i]] = 1;
	}

	(*test_input) = (double**)malloc(TEST_SIZE * sizeof(double*));
	(*test_output) = (double**)malloc(TEST_SIZE * sizeof(double*));
	for(int i = 0; i < TEST_SIZE; i++){

		(*test_input)[i] = (double*)malloc(IMAGE_SIZE * sizeof(double));
		for(int j = 0; j < IMAGE_SIZE; j++)
			(*test_input)[i][j] = (double)test_images[i][j] / 255.0;

		(*test_output)[i] = (double*)calloc(OUTPUT_SIZE, sizeof(double));
		(*test_output)[i][test_labels[i]] = 1;
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
		free((*training_input)[i]);
		free((*training_output)[i]);
	}
	for(int i = 0; i < TEST_SIZE; i++){
		free((*test_input)[i]);
		free((*test_output)[i]);
	}
	free(*training_input);
	free(*training_output);
	free(*test_input);
	free(*test_output);

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
	
	printf("Load complete in %.2lf seconds.\n", time_taken); 
	return;
}
