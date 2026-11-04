# Public dependencies come from the activated cuVS environment.
CUVS_ROOT ?= $(CONDA_PREFIX)
ifeq ($(strip $(CUVS_ROOT)),)
$(error Activate the cuVS conda environment before building quantizers)
endif
CUDA_PATH ?= $(CUVS_ROOT)
NVCC ?= $(CUDA_PATH)/bin/nvcc
HOST_CXX ?= /usr/bin/g++
CUDA_CAPABILITY ?= sm_70
BUILD_DIR ?= ../../build/quantization
PRIVATE_ROOT := ../../third_party/cuvs_rabitq
CUDA_INCLUDE := $(if $(wildcard $(CUDA_PATH)/targets/x86_64-linux/include/cuda_runtime.h),$(CUDA_PATH)/targets/x86_64-linux/include,$(CUDA_PATH)/include)
CUDA_LIB := $(if $(wildcard $(CUDA_PATH)/targets/x86_64-linux/lib/libcudart.so*),$(CUDA_PATH)/targets/x86_64-linux/lib,$(CUDA_PATH)/lib64)
INCLUDES := -I. -I$(CUVS_ROOT)/include -I$(CUVS_ROOT)/include/rapids -I$(CUDA_INCLUDE) -I$(PRIVATE_ROOT)/src/neighbors -I$(PRIVATE_ROOT)/src
FLAGS := --std=c++17 --extended-lambda --expt-relaxed-constexpr -arch=$(CUDA_CAPABILITY) -Wno-deprecated-gpu-targets -DRAFT_SYSTEM_LITTLE_ENDIAN=1 -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -DHIGH_ACC_FAST_SCAN -Xcompiler "-O3 -fPIC -fopenmp -pthread" -ccbin $(HOST_CXX)
LIBS := -L$(CUVS_ROOT)/lib -L$(CUDA_LIB) -lcuvs -lraft -lrmm -lrapids_logger -lcublas -lcublasLt -lcusolver -lcurand -lcudart -lcuda -ldl -lrt -lpthread -lgomp
RPATH := -Xlinker -rpath -Xlinker $(CUVS_ROOT)/lib -Xlinker -rpath -Xlinker $(CUDA_LIB)
PRIVATE_NAMES := initializer_gpu quantizer_gpu quantizer_gpu_fast rotator_gpu
PRIVATE_OBJECTS := $(addprefix $(BUILD_DIR)/,$(addsuffix .o,$(PRIVATE_NAMES)))
OBJECTS := $(PRIVATE_OBJECTS) $(BUILD_DIR)/rabitq_builder.o $(BUILD_DIR)/ivf_gpu_build_only.o $(BUILD_DIR)/pq_builder.o
.PHONY: all check clean
all: $(BUILD_DIR)/pq-builder $(BUILD_DIR)/rabitq-builder
$(BUILD_DIR)/pq-builder: $(BUILD_DIR)/pq_builder.o
	$(NVCC) $(FLAGS) $^ -o $@ $(LIBS) $(RPATH)
$(BUILD_DIR)/rabitq-builder: $(BUILD_DIR)/rabitq_builder.o $(BUILD_DIR)/ivf_gpu_build_only.o $(PRIVATE_OBJECTS)
	$(NVCC) $(FLAGS) $^ -o $@ $(LIBS) $(RPATH)
$(BUILD_DIR)/%.o: %.cu vector_reader.h Makefile
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(FLAGS) $(INCLUDES) -MMD -MP -c $< -o $@
$(BUILD_DIR)/%.o: $(PRIVATE_ROOT)/src/neighbors/ivf_rabitq/gpu_index/%.cu Makefile
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(FLAGS) $(INCLUDES) -MMD -MP -c $< -o $@
$(BUILD_DIR)/vector-reader-test: ../../tests/native/quantization_reader_test.cpp vector_reader.h
	mkdir -p $(BUILD_DIR)
	$(HOST_CXX) -std=c++17 -O2 $< -o $@
check: $(BUILD_DIR)/vector-reader-test
	$(BUILD_DIR)/vector-reader-test
clean:
	rm -f $(OBJECTS) $(OBJECTS:.o=.d) $(BUILD_DIR)/pq-builder $(BUILD_DIR)/rabitq-builder $(BUILD_DIR)/vector-reader-test
-include $(OBJECTS:.o=.d)
