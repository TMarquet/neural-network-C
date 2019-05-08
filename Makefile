all: main

main: main.c
	gcc main.c -o main -lm -g

clean:
	rm main