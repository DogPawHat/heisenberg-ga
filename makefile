CUDA_INSTALL_PATH ?= /usr/local/cuda
BOOST_INCLUDE_PATH ?= /home/ciaran/include

CXX := g++
CC := gcc
LINK := g++ -fPIC
NVCC := nvcc -ccbin /usr/bin

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include -I$(BOOST_INCLUDE_PATH)

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)

LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib -lcudart

BUILD_DIR = build
OBJS =  $(BUILD_DIR)/main.cu.o $(BUILD_DIR)/rand.cu.o $(BUILD_DIR)/algorithmkernel.cu.o
TARGET = $(BUILD_DIR)/heisenburg-ga
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA)

.SUFFIXES: .c .cpp .cu .o

$(BUILD_DIR)/%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS) Makefile
	$(LINKLINE)
