SRCS = mnist.c network.c
OBJS = ${SRCS:.c=.o}
HDS = ${SRCS:.c=.h}

all: main

main: main.c ${OBJS}
	gcc main.c ${OBJS} -o main -lm -g

.c.o: ${SRCS} ${HDS}
	gcc -c $<

clean:
	rm -f main ${OBJS}
