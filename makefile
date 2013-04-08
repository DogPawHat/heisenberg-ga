CUDA_INSTALL_PATH ?= /usr/local/cuda
#
# On windows, store location of Visual Studio compiler
# into the environment. This will be picked up by nvcc,
# even without explicitly being passed.
# On Linux, use whatever gcc is in the current path
# (so leave compiler-bindir undefined):
#
ifdef ON_WINDOWS
export compiler-bindir := c:/mvs/bin
endif
NVCC := nvcc


INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include
export OPENCC_FLAGS :=
export PTXAS_FLAGS :=
CFLAGS := $(OPENCC_FLAGS) $(PTXAS_FLAGS) $(INCLUDES) -arch sm_20
LDFLAGS := -L$(CUDA_INSTALL_PATH)/lib -lcudart
#
# cuda and C/C++ compilation rules, with
# dependency generation:
#

BUILD_DIR = build



$(BUILD_DIR)/%.o : %.cpp
	$(NVCC) -c $< $(CFLAGS) -o $@
	$(NVCC) -M $< $(CFLAGS) > $@.dep
$(BUILD_DIR)/%.o : %.c
	$(NVCC) -c $< $(CFLAGS) -o $@
	$(NVCC) -M $< $(CFLAGS) > $@.dep
$(BUILD_DIR)/%.o : %.cu
	$(NVCC) -c $< $(CFLAGS) -o $@
	$(NVCC) -M $< $(CFLAGS) > $@.dep
#
# Pick up generated dependency files, and
# add /dev/null because gmake does not consider
# an empty list to be a list:
#
include $(wildcard *.dep) /dev/null
#
# Define the application;
# for each object file, there must be a
# corresponding .c or .cpp or .cu file:
#

OBJECTS = $(BUILD_DIR)/main.o $(BUILD_DIR)/rand.o $(BUILD_DIR)/algorithmkernel.o
APP = $(BUILD_DIR)/heisenberg-ga
$(APP) : $(OBJECTS)
	$(NVCC) $(OBJECTS) $(LDFLAGS) -o $@
#
# Cleanup:
#
clean :
	$(RM) $(OBJECTS) *.dep


