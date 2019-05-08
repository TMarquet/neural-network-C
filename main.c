#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#define FILENAME "neural_network_parameters.txt"

/*
Reference: http://neuralnetworksanddeeplearning.com/chap1.html
*/

// Structure for neural network
typedef struct structNetwork{

	int num_layers;
	int* sizes;
	double** biases;
	double*** weights;

} Network;

// Generate uniform random float between -1 and 1
double randFloat(){

	double value = rand();
	return 2.0*value/(double)RAND_MAX - 1.0;
}

// Allocate memory and initialize random parameters in NN
Network* initNetwork(int num_layers, int sizes[]){

	// Allocate memory for struct
	Network* network = (Network*)malloc(sizeof(Network));

	// Set num layers
	network->num_layers = num_layers;

	// Allocate memory for size array, read values
	network->sizes = (int*)malloc(num_layers*sizeof(int));
	for(int i = 0; i < num_layers; i++)
		network->sizes[i] = sizes[i];

	// Allocate memory for biases, fill with random values
	network->biases = (double**)malloc(num_layers*sizeof(double*));
	for(int i = 1; i < num_layers; i++){
		network->biases[i] = (double*)malloc(sizes[i]*sizeof(double));
		for(int j = 0; j < sizes[i]; j++)
			network->biases[i][j] = randFloat();
	}

	// Allocate memory for weights, fill with random values
	network->weights = (double***)malloc(num_layers*sizeof(double**));
	for(int i = 0; i < num_layers-1; i++){
		network->weights[i] = (double**)malloc(sizes[i+1]*sizeof(double*));
		for(int j = 0; j < sizes[i+1]; j++){
			network->weights[i][j] = (double*)malloc(sizes[i]*sizeof(double));
			for(int k = 0; k < sizes[i]; k++)
				network->weights[i][j][k] = randFloat();
		}
	}
	return network;
}

// Squishification function
double sigmoid(double z){

	return 1.0 / (1.0 + exp(-z));
}

// Forward propagate, retrieve output from input
double* feedForward(Network* network, double* input){

	int size_from, size_to;
	double* activation_from;
	double* activation_to;

	for(int i = 0; i < network->num_layers-1; i++){

		size_from = network->sizes[i];
		size_to = network->sizes[i+1];

		if (i == 0){
			activation_from = (double*)malloc(size_from*sizeof(double));
			memcpy(activation_from, input, size_from*sizeof(double));
			activation_to = (double*)malloc(size_to*sizeof(double));
		}
		else{
			activation_from = (double*)realloc(activation_from, size_from*sizeof(double));
			memcpy(activation_from, activation_to, size_from*sizeof(double));
			activation_to = (double*)realloc(activation_to, size_to*sizeof(double));
		}

		for(int j = 0; j < size_to; j++){
			activation_to[j] = 0;

			for(int k = 0; k < size_from; k++)
				activation_to[j] += network->weights[i][j][k] * activation_from[k];

			activation_to[j] += network->biases[i+1][j];
			activation_to[j] = sigmoid(activation_to[j]);
		}
	}
	free(activation_from);
	return activation_to;
}

// Save parameters of NN to file
void saveNetwork(Network* network, char* filename){

	// Open file
	FILE* fptr = fopen(filename, "w+");
	if (!fptr){
		printf("File %s not found.\n", filename);
		return;
	}

	// Save num_layers
	fprintf(fptr, "num_layers = %d\n\n", network->num_layers);

	// Save sizes
	fprintf(fptr, "sizes = ");
	for(int i = 0; i < network->num_layers; i++)
		fprintf(fptr, "%d ", network->sizes[i]);
	fprintf(fptr, "\n\n");

	// Save biases
	fprintf(fptr, "biases = \n\n");
	for(int i = 1; i < network->num_layers; i++){
		for(int j = 0; j < network->sizes[i]; j++)
			fprintf(fptr, "%lf ", network->biases[i][j]);
		fprintf(fptr, "\n\n");
	}

	// Save weights
	fprintf(fptr, "weights = \n\n");
	for(int i = 0; i < network->num_layers-1; i++){
		for(int j = 0; j < network->sizes[i+1]; j++){
			for(int k = 0; k < network->sizes[i]; k++)
				fprintf(fptr, "%lf ", network->weights[i][j][k]);
			fprintf(fptr, "\n");
		}
		fprintf(fptr, "\n");
	}

	fclose(fptr);
	return;
}

// Load parameters to NN
Network* loadNetwork(Network* network, char* filename){

	// Open file
	FILE* fptr = fopen(filename, "r");
	if (!fptr){
		printf("File %s not found.\n", filename);
		return NULL;
	}

	// Allocate memory for struct
	network = (Network*)malloc(sizeof(Network));

	// Load num_layers
	fscanf(fptr, "num_layers = %d\n\n", &(network->num_layers));

	// Allocate mem, load sizes
	network->sizes = (int*)malloc(network->num_layers*sizeof(int));
	fscanf(fptr, "sizes = ");
	for(int i = 0; i < network->num_layers; i++)
		fscanf(fptr, "%d ", &(network->sizes[i]));
	fscanf(fptr, "\n\n");

	// Allocate mem, load biases
	network->biases = (double**)malloc(network->num_layers*sizeof(double*));
	fscanf(fptr, "biases = \n\n");
	for(int i = 1; i < network->num_layers; i++){
		network->biases[i] = (double*)malloc(network->sizes[i]*sizeof(double));
		for(int j = 0; j < network->sizes[i]; j++)
			fscanf(fptr, "%lf ", &(network->biases[i][j]));
		fscanf(fptr, "\n\n");
	}

	// Allocate mem, load weights
	network->weights = (double***)malloc(network->num_layers*sizeof(double**));
	fscanf(fptr, "weights = \n\n");
	for(int i = 0; i < network->num_layers-1; i++){
		network->weights[i] = (double**)malloc(network->sizes[i+1]*sizeof(double*));
		for(int j = 0; j < network->sizes[i+1]; j++){
			network->weights[i][j] = (double*)malloc(network->sizes[i]*sizeof(double));
			for(int k = 0; k < network->sizes[i]; k++)
				fscanf(fptr, "%lf ", &(network->weights[i][j][k]));
			fscanf(fptr, "\n");
		}
		fscanf(fptr, "\n");
	}

	fclose(fptr);
	return network;
}

// The main attraction
int main(){

	/*
	srand(time(NULL));

	int num_layers = 3;
	int sizes[] = {4, 3, 2};

	Network* network = initNetwork(num_layers, sizes);
	saveNetwork(network, FILENAME);
	*/

	Network* network = loadNetwork(network, FILENAME);
	double input_vector[4] = {4.0, 3.0, 2.5, 3.4};
	double* output = feedForward(network, input_vector);

	for(int i = 0; i < network->sizes[network->num_layers-1]; i++)
		printf("Value: %lf\n", output[i]);

	return 0;
}






/*
network.py
~~~~~~~~~~



class Network(object):

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

*/





// MY FUNCTIONS THAT AREN'T BEING USED

/*
// Elementwise sigmoid operation on (Ax1)
double* elementwiseSigmoid(double* z, int dim_A){

	double* output = (double*)malloc(dim_A*sizeof(double));

	for(int i = 0; i < dim_A; i++)
		output[i] = sigmoid(z[i]);
	
	return output;
}

// Matrix multiplication for (AxB) times (Bx1)
double* matVecProd(double** matrix, double* vector, int dim_A, int dim_B){

	double* product = (double*)malloc(dim_A*sizeof(double));

	for(int i = 0; i < dim_A; i++){
		product[i] = 0;

		for(int j = 0; j < dim_B; j++)
			product[i] += matrix[i][j] * vector[j];
	}
	return product;
}

// Vector sum (aX1)
double* vecVecSum(double* vector_1, double* vector_2, int dim_A){

	double* sum = (double*)malloc(dim_A*sizeof(double));

	for(int i = 0; i < dim_A; i++)
		sum[i] = vector_1[i] + vector_2[i];
	
	return sum;
}
*/