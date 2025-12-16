# LOS Feedback Controller Reference Implementation
# Builds the C reference test harness for comparison with Rust implementation

CC = gcc
CFLAGS = -O2 -Wall -Wextra

TARGET = los_test
SOURCES = main.c CF_LOS_FB_40Hz.c
HEADERS = CF_LOS_FB_40Hz.h

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $@ $(SOURCES) -lm

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
