all: main

main: main.c
	gcc main.c -o main -lm -g

python:
	python3 load_data_set.py

clean:
	rm -f main neural_network_parameters.txt data/*
