#define TRAINING_LABEL_NAME "data/train-labels-idx1-ubyte"
#define TRAINING_IMAGE_NAME "data/train-images-idx3-ubyte"
#define TEST_LABEL_NAME "data/t10k-labels-idx1-ubyte"
#define TEST_IMAGE_NAME "data/t10k-images-idx3-ubyte"

#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

// Load MNIST training and test data into unsigned char arrays
void loadMNIST(unsigned char**, unsigned char***, unsigned char**, unsigned char ***);

// Convert raw MNIST data into double arrays suited for training a NN
void convertMNIST(unsigned char**, unsigned char***, unsigned char**, unsigned char***,
					double***, double***, double***, double***);

// Free the MNIST double arrays
void freeMNIST(double***, double***, double***, double***);

// Show some examples of imported training data
void showMNIST(double**, double**, int);
