#define PARAM_NAME 	"neural_network_parameters.txt"
#define LAST 		network->num_layers-1

// Structure for neural network
typedef struct structNetwork{

	int 		num_layers;
	int* 		sizes;
	double** 	biases;
	double*** 	weights;

} Network;

// Allocate memory and initialize random parameters in NN
Network* initNetwork(int, int[*]);

// Save parameters of NN to file
void saveNetwork(Network*, char*);

// Allocate memory and load parameters to NN
Network* loadNetwork(char*);

// Forward propagate, retrieve output from input
double* feedForward(Network*, double*);

// Training algorithm
void stochasticGradientDescent(Network*, double**, double**, int, int, int, double);

// Applying SGD for the mini batch
void update_mini_batch(Network*, double**, double**, int, double);

// Determine the gradient of the NN
void backPropagation(Network*, double*, double*, double***, double****);

// Test how well the NN does for given test data
void evaluate(Network*, double**, double**, int);

// Squishification function
double sigmoid(double);

// Derivative of squishification function
double sigmoidPrime(double);

// Generate uniform random float between -1 and 1
double randFloat();

// Run benchmark to test current training functions
void benchmarkSGD(Network*, double**, double**, int, int, int, double);
