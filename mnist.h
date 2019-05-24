#define TRAINING_LABEL_NAME "data/train-labels-idx1-ubyte"
#define TRAINING_IMAGE_NAME "data/train-images-idx3-ubyte"
#define TEST_LABEL_NAME "data/t10k-labels-idx1-ubyte"
#define TEST_IMAGE_NAME "data/t10k-images-idx3-ubyte"

#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

// Load MNIST training data into arrays suited for training a NN
void loadMNIST(double***, double***, double***, double***);

// Free the MNIST arrays
void freeMNIST(double***, double***, double***, double***);

// Show some examples of imported training data
void showMNIST(double**, double**, int);

// Run benchmark to test load speeds
void benchmarkLoadMNIST(double***, double***, double***, double***);
