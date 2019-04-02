CC = gcc

CFLAGS  = -g -Wall -Wfatal-errors -O3

# add this to compile sparse version
CFLAGS += -D SPARSE

# remove this to compile debug version (slower but with some checks for errors)
CFLAGS += -D NDEBUG

CFILES = subproblem.c queue.c sspDense.c sspSparse.c murtysplitDense.c murtysplitSparse.c da.c

#TARGET = mhtda
# $(TARGET): $(CFILES)
#	$(CC) $(CFLAGS) -o $(TARGET) $(CFILES)

SLIB = mhtda.so
$(SLIB): $(CFILES)
	$(CC) $(CFLAGS) -shared -fPIC -o $(SLIB) $(CFILES)

all: $(SLIB)

clean:
	rm $(SLIB)
