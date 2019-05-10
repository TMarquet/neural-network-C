'''
This is not my original code
Modified from
https://github.com/MichalDanielDobrzanski/DeepLearningPython35/blob/master/mnist_loader.py
'''

import pickle
import gzip
import numpy as np

def load_data():

	f = gzip.open('mnist.pkl.gz', 'rb')
	tr_d, va_d, te_d = pickle.load(f, encoding="latin1")
	f.close()

	training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
	training_results = [vectorized_result(y) for y in tr_d[1]]

	validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
	validation_results = [vectorized_result(y) for y in va_d[1]]

	test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
	test_results = [vectorized_result(y) for y in te_d[1]]

	return training_inputs, training_results, validation_inputs, validation_results, test_inputs, test_results

def vectorized_result(j):

	e = np.zeros((10, 1))
	e[j] = 1.0
	return e

tr_i, tr_o, va_i, va_o, te_i, te_o = load_data()


training_input = np.zeros((len(tr_i), 784))

for i in range(len(tr_i)):
	for j in range(784):
		training_input[i,j] = tr_i[i][j]

np.savetxt('data/training_input.txt', training_input, fmt='%1.2e', delimiter=',')

print("1/6 completed.")

training_output = np.zeros((len(tr_o), 10))

for i in range(len(tr_o)):
	for j in range(10):
		training_output[i,j] = tr_o[i][j]

np.savetxt('data/training_output.txt', training_output, fmt='%1.2e', delimiter=',')

print("2/6 completed.")

validation_input = np.zeros((len(va_i), 784))

for i in range(len(va_i)):
	for j in range(784):
		validation_input[i,j] = va_i[i][j]

np.savetxt('data/validation_input.txt', validation_input, fmt='%1.2e', delimiter=',')

print("3/6 completed.")

validation_output = np.zeros((len(va_o), 10))

for i in range(len(va_o)):
	for j in range(10):
		validation_output[i,j] = va_o[i][j]

np.savetxt('data/validation_output.txt', validation_output, fmt='%1.2e', delimiter=',')

print("4/6 completed.")

test_input = np.zeros((len(te_i), 784))

for i in range(len(te_i)):
	for j in range(784):
		test_input[i,j] = te_i[i][j]

np.savetxt('data/test_input.txt', test_input, fmt='%1.2e', delimiter=',')

print("5/6 completed.")

test_output = np.zeros((len(te_o), 10))

for i in range(len(te_o)):
	for j in range(10):
		test_output[i,j] = te_o[i][j]

np.savetxt('data/test_output.txt', test_output, fmt='%1.2e', delimiter=',')

print("6/6 completed.")

print("Training set size:")
print(len(tr_i))
print("Validation set size:")
print(len(va_i))
print("Test set size:")
print(len(te_i))
