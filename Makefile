NVCC ?= nvcc
GCC ?= g++

ARCH := 80
BUILD_DIR ?=./build

# Find all source files
EXCLUDED_DIRS := scratch build scripts tests logs  # Add the directories you want to exclude here
CU_FILES := $(shell find . -name '*.cu' $(addprefix -not -path "./", $(addsuffix "/*", $(EXCLUDED_DIRS))))
CPP_FILES := $(shell find . -name '*.cpp' $(addprefix -not -path "./", $(addsuffix "/*", $(EXCLUDED_DIRS))))

# Define object files for both cu and cpp
CU_OBJ_FILES := $(patsubst %.cu,$(BUILD_DIR)/obj/%.cu.o,$(notdir $(CU_FILES)))
CPP_OBJ_FILES := $(patsubst %.cpp,$(BUILD_DIR)/obj/%.cpp.o,$(CPP_FILES))

# cuda flags
CUDAFLAGS ?= -g -Xcompiler -fopenmp -lineinfo -O3 -arch=sm_$(ARCH) -gencode=arch=compute_$(ARCH),code=sm_$(ARCH) \
						-gencode=arch=compute_$(ARCH),code=compute_$(ARCH)
CUDAINC	?= -I/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc/include

LDIR_CUDA ?= -L/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc/lib64
LDFLAGS_CUDA ?= -lcudart -lgomp

CPPFLAGS ?= -O3
CPPINC ?= -I${GUROBI_HOME}/include

LDIR_CPP ?= -L${GUROBI_HOME}/lib
LDFLAGS_CPP ?= -lgurobi_c++ -lgurobi100 -lm -D_GLIBCXX_USE_CXX11_ABI=0

all: $(BUILD_DIR)/main.exe

$(BUILD_DIR)/main.exe: $(CU_OBJ_FILES) $(CPP_OBJ_FILES)
	$(NVCC) -o $@ $(CU_OBJ_FILES) $(LDIR_CUDA) $(LDFLAGS_CUDA) $(CPP_OBJ_FILES) $(LDIR_CPP) $(LDFLAGS_CPP)

# Pattern rule for cu files
$(BUILD_DIR)/obj/%.cu.o: %.cu
	mkdir -p $(BUILD_DIR)/obj/
	@echo cu obj files are: $(CU_OBJ_FILES)
	@echo cu files are: $(CU_FILES)
	$(NVCC) $(CUDAFLAGS) $(CUDAINC) -c $< -o $@ $(LDIR_CUDA) $(LDFLAGS_CUDA)


# Pattern rule for cpp files
$(BUILD_DIR)/obj/%.cpp.o: %.cpp
	@mkdir -p $(BUILD_DIR)/obj/$(dir $<) 
	@echo cpp obj files are: $(CPP_OBJ_FILES)
	@echo cpp files are: $(CPP_FILES)
	$(GCC) $(CPPFLAGS) $(CPPINC) -c $< -o $@ $(LDIR_CPP) $(LDFLAGS_CPP)


clean:
	$(RM) -r $(BUILD_DIR)
	@echo SM_VALUE IS $(ARCH)