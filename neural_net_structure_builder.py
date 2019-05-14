import numpy as np 
import matplotlib.pyplot as plt 

layer_sizes = np.array([8, 5, 8, 4])

for i in range(len(layer_sizes)-1):

	start_x = i
	start_y = layer_sizes[i] / 2 + 0.5

	for j in range(layer_sizes[i]):

		start_y -= 1

		end_x = i+1
		end_y = layer_sizes[i+1] / 2 + 0.5

		for k in range(layer_sizes[i+1]):

			end_y -= 1
			plt.plot([start_x, end_x], [start_y, end_y], 'r', linewidth=0.5)

for i in range(len(layer_sizes)):

	x = i
	y = layer_sizes[i] / 2 + 0.5

	for j in range(layer_sizes[i]):

		y -= 1
		plt.plot([x, x], [y, y], 'o', markersize=15, markeredgecolor='k', markerfacecolor='w')

plt.title('Neural Network Structure', fontsize=20)
plt.axis('off')
plt.savefig('neural_net_structure.pdf')
