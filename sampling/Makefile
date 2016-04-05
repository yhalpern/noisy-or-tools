CC=g++
CFLAGS=-I

all: read_ss.o statistics.o sample_ss.o sampling.o mtrand.o
	$(CC) -o read_ss -O3 read_ss.o statistics.o mtrand.o 
	$(CC) -o sample_ss -O3 sample_ss.o sampling.o mtrand.o

reader: read_ss.o statistics.o mtrand.o
	$(CC) -o read_ss -O3 read_ss.o statistics.o mtrand.o

sampler: sample_ss.o sampling.o mtrand.o
	$(CC) -o sample_ss -O3 sample_ss.o sampling.o mtrand.o

clean:
	rm *.o
