.DEFAULT_GOAL := exe
ifeq ($(origin CUDA_PATH), undefined)
CUDA_PATH :=
endif

CUDA_ROOT := $(patsubst %/,%,$(CUDA_PATH))
ifneq ($(wildcard $(CUDA_ROOT)/nvcc),)
CUDA_ROOT := $(patsubst %/,%,$(dir $(CUDA_ROOT)))
endif

NVCC ?= $(shell command -v nvcc 2>/dev/null)
ifeq ($(strip $(NVCC)),)
ifneq ($(strip $(CUDA_ROOT)),)
NVCC := $(CUDA_ROOT)/bin/nvcc
else
NVCC := /usr/local/cuda/bin/nvcc
endif
endif

ifeq ($(strip $(CUDA_ROOT)),)
CUDA_ROOT := $(shell dirname "$$(dirname "$(NVCC)")")
endif

CUDA_TOOLKIT_INCLUDE ?= $(if $(wildcard $(CUDA_ROOT)/targets/x86_64-linux/include/cuda_runtime.h),$(CUDA_ROOT)/targets/x86_64-linux/include,$(CUDA_ROOT)/include)
CUDA_TOOLKIT_LIB_PATH ?= $(if $(wildcard $(CUDA_ROOT)/targets/x86_64-linux/lib/libcudart.so*),$(CUDA_ROOT)/targets/x86_64-linux/lib,$(if $(wildcard $(CUDA_ROOT)/lib64/libcudart.so*),$(CUDA_ROOT)/lib64,$(CUDA_ROOT)/lib))

DEPENDENCY_PREFIX ?= $(if $(CONDA_PREFIX),$(CONDA_PREFIX),$(CUDA_ROOT))
ROOT_DIR := ..
OBJ_DIR := $(ROOT_DIR)/build/obj
TEST_BIN_DIR := $(ROOT_DIR)/build/tests
PROJECT_CPPFLAGS := -I. -I$(ROOT_DIR)/include
LIB_PATH ?= $(ROOT_DIR)
HOST_CXX := $(firstword $(wildcard /usr/bin/g++ /bin/g++) $(shell command -v g++ 2>/dev/null) g++)
HOST_CC := $(firstword $(wildcard /usr/bin/gcc /bin/gcc) $(shell command -v gcc 2>/dev/null) gcc)
HOST_MULTIARCH ?= $(shell $(HOST_CXX) $(PROJECT_CPPFLAGS) -print-multiarch 2>/dev/null)
HOST_MULTIARCH_FALLBACK := $(if $(strip $(HOST_MULTIARCH)),$(HOST_MULTIARCH),x86_64-linux-gnu)
SYSTEM_INCLUDE ?= /usr/include
MINICONDA_INCLUDE ?= $(if $(CONDA_PREFIX),$(CONDA_PREFIX)/include,/usr/include)
SYSTEM_MULTIARCH_INCLUDE ?= /usr/include/$(HOST_MULTIARCH_FALLBACK)
SYSTEM_MULTIARCH_LIB_PATH ?= /usr/lib/$(HOST_MULTIARCH_FALLBACK)
GDRAPI_INCLUDE ?= /usr/local/include
GDRAPI_LIB_PATH ?= /usr/local/lib

ENABLE_CUVS_BUILD ?= 1
PQ_MULTI_QUERY ?= 1
PQ_WARPS_PER_QUERY ?= 1
PQ_EXPERIMENTAL_COUPLED_MULTIWARP ?= 0
PQ_MAX_REGISTERS ?=
PQ_NATIVE_DETAILED_TIMING ?= 1
PQ_NATIVE_COOPERATIVE_PICKUP ?= 1
PQ_NATIVE_COOPERATIVE_FRONTIER_UPDATE ?= 0
PQ_NATIVE_MAX_HEAP_FRONTIER ?= 0
PQ_NATIVE_PROFILE_MODE ?= 0
PQ_FINITE_PERHOP_TIMING ?= 0
PQ_REQUEST_PUBLISH_TIMING ?= 0
RABITQ_MULTI_QUERY ?= 1
RABITQ_MAX_REGISTERS ?=
RABITQ_NATIVE_DETAILED_TIMING ?= 1
RABITQ_NATIVE_COOPERATIVE_PICKUP ?= 1
RABITQ_PHASE_SNAPSHOT_PROFILE ?= 0
GPU_PREDICATE_PARALLEL ?= 1
GPU_PREDICATE_CSR_THRESHOLD ?= 0
FULL_GPU_COOPERATIVE_PICKUP ?= 1
FULL_GPU_UINT8_DP4A ?= 1
CUVS_INCLUDE ?= $(DEPENDENCY_PREFIX)/include
RAFT_INCLUDE ?= $(DEPENDENCY_PREFIX)/include/rapids
CUVS_LIB_PATH ?= $(DEPENDENCY_PREFIX)/lib
CUVS_LIBS ?= -lcuvs -lraft -lrmm -lrapids_logger

CUDA_CAPABILITY ?= sm_70
CUDA_SOURCES := app/main.cu search/pq/backend_pq.cu index/graph_builder.cu search/rabitq/backend_rabitq.cu search/full_gpu/backend_full_gpu.cu
CUDA_OBJECTS := $(addprefix $(OBJ_DIR)/,$(CUDA_SOURCES:.cu=.o))
TARGET := go-fann
HOST_SOURCES := profiling/expansion_trace.cpp data/dataloader.cpp execution/async_io.cpp app/config_loader.cpp index/graph_file_format.cpp index/range_posting_index.cpp
HOST_SOURCES += execution/graph_prefetch.cpp
HOST_OBJECTS := $(addprefix $(OBJ_DIR)/,$(HOST_SOURCES:.cpp=.o))
DATALOADER_TEST_SOURCES := $(HOST_SOURCES)
HOST_CPPFLAGS := $(PROJECT_CPPFLAGS) -I$(MINICONDA_INCLUDE) -I$(SYSTEM_INCLUDE) -I$(SYSTEM_MULTIARCH_INCLUDE)
# Recursive header prerequisites conservatively rebuild all consumers on changes.
rwildcard = $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))
SOURCE_HEADERS := $(call rwildcard,./,*.h) $(call rwildcard,./,*.cuh) $(call rwildcard,./,*.hpp) $(call rwildcard,$(ROOT_DIR)/include/,*.h)

RUNTIME_DEFAULTS_TEST := ../tests/native/runtime_defaults_test.cpp
PQ_QUERY_WORKER_MODE_CONFIG_TEST := ../tests/native/pq_query_worker_mode_config_test.cpp
PQ_QUERY_WORKER_MODE_SOURCE_CONTRACT_TEST := ../tests/native/pq_query_worker_mode_source_contract_test.py
PQ_DEVICE_RESIDENT_SOURCE_CONTRACT_TEST := ../tests/native/pq_device_resident_source_contract_test.py
PQ_REQUEST_RING_SOURCE_CONTRACT_TEST := ../tests/native/pq_request_ring_source_contract_test.py
PQ_FINITE_SCHEDULER_TEST := ../tests/native/pq_finite_scheduler_test.cpp
DURABLE_CSV_TEST := ../tests/native/durable_csv_test.cpp
RERANK_VECTOR_STORE_TEST := ../tests/native/rerank_vector_store_test.cpp
RERANK_VECTOR_STORAGE_CONFIG_TEST := ../tests/native/rerank_vector_storage_config_test.cpp
RABITQ_MULTIQUERY_SOURCE_CONTRACT_TEST := ../tests/native/rabitq_multiquery_source_contract_test.py
RABITQ_LOWFREQ_MERGE_TEST := ../tests/native/rabitq_lowfreq_merge_test.cpp
POSTING_SHORTLIST_HEAP_TEST := ../tests/native/posting_shortlist_heap_test.cpp
RABITQ_FINITE_CONFIG_TEST := ../tests/native/rabitq_finite_config_test.cpp
RABITQ_FINITE_SOURCE_CONTRACT_TEST := ../tests/native/rabitq_finite_source_contract_test.py
RABITQ_MAILBOX_RTT_SOURCE_CONTRACT_TEST := ../tests/native/rabitq_mailbox_rtt_source_contract_test.py
RABITQ_PROFILE_SOURCE_CONTRACT_TEST := ../tests/native/rabitq_profile_source_contract_test.py
RABITQ_FLOAT_GPU_RERANK_SOURCE_CONTRACT_TEST := ../tests/native/rabitq_float_gpu_rerank_source_contract_test.py
RANGE_POSTING_INDEX_TEST := ../tests/native/range_posting_index_test.cpp
RANGE_POSTING_GENERATOR_TEST := tests.python.range_posting_generator_test
RANGE_HIERARCHY_GENERATOR_TEST := tests.python.range_hierarchy_generator_test
GRAPH_BUILDER_MULTIWORKER_SOURCE_CONTRACT_TEST := ../tests/native/graph_builder_multiworker_source_contract_test.py
RANGE_POSTING_CONFIG_TEST := ../tests/native/range_posting_config_test.cpp
RANGE_ACCESS_PLANNER_TEST := ../tests/native/range_access_planner_test.cpp
RANGE_POSTING_SOURCE_CONTRACT_TEST := ../tests/native/range_posting_source_contract_test.py
RABITQ_CPU_WORKER_BUDGET_TEST := ../tests/native/rabitq_cpu_worker_budget_test.cpp
GRAPH_ROUTE_SELECTION_TEST := ../tests/native/graph_route_selection_test.cpp
CANDIDATE_STREAM_SOURCE_CONTRACT_TEST := ../tests/native/candidate_stream_source_contract_test.py
FULL_GPU_OPTIMIZATION_SOURCE_CONTRACT_TEST := ../tests/native/full_gpu_optimization_source_contract_test.py
RANGE_POSTING_MERGE_TEST := ../tests/native/range_posting_merge_test.cpp
OPERATOR_FUZZY_REWRITE_TEST := ../tests/native/operator_fuzzy_rewrite_test.cpp
OR_ROUTE_PLANNER_TEST := ../tests/native/or_route_planner_test.cpp

NVCC_FLAGS := --std=c++17 --extended-lambda -DRAFT_SYSTEM_LITTLE_ENDIAN=1 -Xcompiler "-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -fno-strict-aliasing -pthread -O2 -DNDEBUG -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=2" -Wno-deprecated-gpu-targets
PQ_NVCC_FLAGS := -DGO_FANN_PQ_MULTI_QUERY=$(PQ_MULTI_QUERY) -DGO_FANN_PQ_WARPS_PER_QUERY=$(PQ_WARPS_PER_QUERY) -DGO_FANN_PQ_EXPERIMENTAL_COUPLED_MULTIWARP=$(PQ_EXPERIMENTAL_COUPLED_MULTIWARP) -DGO_FANN_PQ_NATIVE_DETAILED_TIMING=$(PQ_NATIVE_DETAILED_TIMING) -DGO_FANN_PQ_NATIVE_COOPERATIVE_PICKUP=$(PQ_NATIVE_COOPERATIVE_PICKUP)
PQ_NVCC_FLAGS += -DGO_FANN_PQ_NATIVE_COOPERATIVE_FRONTIER_UPDATE=$(PQ_NATIVE_COOPERATIVE_FRONTIER_UPDATE)
PQ_NVCC_FLAGS += -DGO_FANN_PQ_NATIVE_MAX_HEAP_FRONTIER=$(PQ_NATIVE_MAX_HEAP_FRONTIER)
PQ_NVCC_FLAGS += -DGO_FANN_PQ_NATIVE_PROFILE_MODE=$(PQ_NATIVE_PROFILE_MODE)
PQ_NVCC_FLAGS += -DGO_FANN_PQ_FINITE_PERHOP_TIMING=$(PQ_FINITE_PERHOP_TIMING)
PQ_NVCC_FLAGS += -DGO_FANN_PQ_REQUEST_PUBLISH_TIMING=$(PQ_REQUEST_PUBLISH_TIMING)
PQ_NVCC_FLAGS += -DGO_FANN_GPU_PREDICATE_PARALLEL=$(GPU_PREDICATE_PARALLEL)
PQ_NVCC_FLAGS += -DGO_FANN_GPU_PREDICATE_CSR_THRESHOLD=$(GPU_PREDICATE_CSR_THRESHOLD)
RABITQ_NVCC_FLAGS := -DGO_FANN_RABITQ_MULTI_QUERY=$(RABITQ_MULTI_QUERY) -DGO_FANN_RABITQ_NATIVE_DETAILED_TIMING=$(RABITQ_NATIVE_DETAILED_TIMING) -DGO_FANN_RABITQ_NATIVE_COOPERATIVE_PICKUP=$(RABITQ_NATIVE_COOPERATIVE_PICKUP)
RABITQ_NVCC_FLAGS += -DGO_FANN_RABITQ_PHASE_SNAPSHOT_PROFILE=$(RABITQ_PHASE_SNAPSHOT_PROFILE)
RABITQ_NVCC_FLAGS += -DGO_FANN_GPU_PREDICATE_PARALLEL=$(GPU_PREDICATE_PARALLEL)
RABITQ_NVCC_FLAGS += -DGO_FANN_GPU_PREDICATE_CSR_THRESHOLD=$(GPU_PREDICATE_CSR_THRESHOLD)
FULL_GPU_NVCC_FLAGS := -DGO_FANN_FULL_GPU_COOPERATIVE_PICKUP=$(FULL_GPU_COOPERATIVE_PICKUP)
FULL_GPU_NVCC_FLAGS += -DGO_FANN_FULL_GPU_UINT8_DP4A=$(FULL_GPU_UINT8_DP4A)
# Optional overlap-graph support is active in the Full-GPU backend;
# keep its sorted-frontier and overlap headers in the dependency set.

ifneq ($(strip $(PQ_MAX_REGISTERS)),)
PQ_NVCC_FLAGS += --maxrregcount=$(PQ_MAX_REGISTERS)
endif

ifneq ($(strip $(RABITQ_MAX_REGISTERS)),)
RABITQ_NVCC_FLAGS += --maxrregcount=$(RABITQ_MAX_REGISTERS)
endif

ifeq ($(ENABLE_CUVS_BUILD),1)
NVCC_FLAGS += -DGO_FANN_ENABLE_CUVS_BUILD -I$(CUVS_INCLUDE) -I$(RAFT_INCLUDE) -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
LINK_CUVS := $(if $(strip $(CUVS_LIB_PATH)),-L$(CUVS_LIB_PATH),) $(CUVS_LIBS)
RPATH_CUVS :=
ifneq ($(strip $(CUVS_LIB_PATH)),)
RPATH_CUVS := -Wl,-rpath=$(CUVS_LIB_PATH)
endif
else
LINK_CUVS :=
RPATH_CUVS :=
endif

.PHONY: all exe run clean runtime-defaults-test  pq-query-worker-mode-config-test pq-query-worker-mode-source-contract-test pq-device-resident-source-contract-test pq-request-ring-source-contract-test pq-finite-scheduler-test durable-csv-test  rerank-vector-store-test rerank-vector-storage-config-test   rabitq-multiquery-source-contract-test rabitq-lowfreq-merge-test posting-shortlist-heap-test rabitq-finite-config-test rabitq-finite-source-contract-test rabitq-mailbox-rtt-source-contract-test rabitq-profile-source-contract-test rabitq-float-gpu-rerank-source-contract-test range-posting-index-test range-posting-generator-test range-hierarchy-generator-test graph-builder-multiworker-source-contract-test range-posting-config-test range-access-planner-test range-posting-source-contract-test  rabitq-cpu-worker-budget-test runtime-route-selection-test candidate-stream-source-contract-test full-gpu-optimization-source-contract-test range-posting-merge-test operator-fuzzy-rewrite-test or-route-planner-test

$(OBJ_DIR)/search/pq/backend_pq.o: NVCC_FLAGS += $(PQ_NVCC_FLAGS)
$(OBJ_DIR)/search/rabitq/backend_rabitq.o: NVCC_FLAGS += $(RABITQ_NVCC_FLAGS)
$(OBJ_DIR)/search/full_gpu/backend_full_gpu.o: NVCC_FLAGS += $(FULL_GPU_NVCC_FLAGS)
# Keep flag changes from silently reusing an object compiled in a different mode.
# The .o suffix makes this generated stamp follow the existing object ignore rule.
.PHONY: full-gpu-flags-force
$(OBJ_DIR)/flags/full_gpu.o: full-gpu-flags-force
	@mkdir -p $(@D)
	@printf '%s\n' '$(FULL_GPU_NVCC_FLAGS)' > $@.tmp
	@cmp -s $@ $@.tmp || cp $@.tmp $@
	@rm -f $@.tmp
$(OBJ_DIR)/search/full_gpu/backend_full_gpu.o: $(OBJ_DIR)/flags/full_gpu.o search/full_gpu/full_gpu_sorted_frontier.cuh search/full_gpu/full_gpu_overlap_types.cuh search/full_gpu/full_gpu_overlap_search.cuh

.PHONY: pq-flags-force
$(OBJ_DIR)/flags/pq.o: pq-flags-force
	@mkdir -p $(@D)
	@printf '%s\n' '$(PQ_NVCC_FLAGS)' > $@.tmp
	@cmp -s $@ $@.tmp || cp $@.tmp $@
	@rm -f $@.tmp
$(OBJ_DIR)/search/pq/backend_pq.o: $(OBJ_DIR)/flags/pq.o

.PHONY: rabitq-flags-force
$(OBJ_DIR)/flags/rabitq.o: rabitq-flags-force
	@mkdir -p $(@D)
	@printf '%s\n' '$(RABITQ_NVCC_FLAGS)' > $@.tmp
	@cmp -s $@ $@.tmp || cp $@.tmp $@
	@rm -f $@.tmp
$(OBJ_DIR)/search/rabitq/backend_rabitq.o: $(OBJ_DIR)/flags/rabitq.o

$(OBJ_DIR)/%.o: %.cu $(SOURCE_HEADERS)
	@mkdir -p $(@D)
	$(NVCC) $(PROJECT_CPPFLAGS) -c $(NVCC_FLAGS) -O3 -Xcompiler "-fPIC -fopenmp" --ptxas-options=-v -arch=$(CUDA_CAPABILITY) $< -o $@ \
		-ccbin $(HOST_CXX) -I. -I$(ROOT_DIR)/include -I$(DEPENDENCY_PREFIX)/include -I$(CUDA_TOOLKIT_INCLUDE) -I$(SYSTEM_INCLUDE) -I$(MINICONDA_INCLUDE) -I$(SYSTEM_MULTIARCH_INCLUDE) -I$(GDRAPI_INCLUDE) -I$(LIB_PATH) -lcurand

$(OBJ_DIR)/%.o: %.cpp $(SOURCE_HEADERS)
	@mkdir -p $(@D)
	$(HOST_CXX) $(PROJECT_CPPFLAGS) -c -O3 -fPIC -fopenmp $< $(HOST_CPPFLAGS) -std=c++17 -o $@

all: exe

.PHONY: rabitq-query-rotation-test rabitq-query-rotation-config-test
rabitq-query-rotation-test:
	@tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(NVCC) $(PROJECT_CPPFLAGS) -std=c++17 -O3 -arch=$(CUDA_CAPABILITY) ../tests/native/rabitq_query_rotation_test.cu -lcublas -o "$$tmp" && "$$tmp"

rabitq-query-rotation-config-test:
	@tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -std=c++17 $(HOST_CPPFLAGS) ../tests/native/rabitq_query_rotation_config_test.cpp app/config_loader.cpp -o "$$tmp" && "$$tmp"

exe: $(CUDA_OBJECTS) $(HOST_OBJECTS)
	$(HOST_CXX) $(PROJECT_CPPFLAGS) -o $(TARGET) $(CUDA_OBJECTS) $(HOST_OBJECTS) \
		-L$(LIB_PATH) -Wl,-rpath=$(LIB_PATH) -L$(DEPENDENCY_PREFIX)/lib -Wl,-rpath=$(DEPENDENCY_PREFIX)/lib -L$(CUDA_TOOLKIT_LIB_PATH) -Wl,-rpath=$(CUDA_TOOLKIT_LIB_PATH) \
		-L$(GDRAPI_LIB_PATH) -Wl,-rpath=$(GDRAPI_LIB_PATH) -L$(SYSTEM_MULTIARCH_LIB_PATH) -Wl,-rpath=$(SYSTEM_MULTIARCH_LIB_PATH) \
		-lcudart -lcublas -lgdrapi -lcrypto \
		$(LINK_CUVS) $(RPATH_CUVS) -lrt -lpthread -ldl -lcuda -lgomp -fopenmp

runtime-defaults-test:
	@tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra -std=c++17 $(RUNTIME_DEFAULTS_TEST) -o "$$tmp"; \
		"$$tmp"

pq-query-worker-mode-config-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 \
			$(PQ_QUERY_WORKER_MODE_CONFIG_TEST) app/config_loader.cpp -o "$$tmp"; \
		"$$tmp"

pq-query-worker-mode-source-contract-test:
	python3 $(PQ_QUERY_WORKER_MODE_SOURCE_CONTRACT_TEST)

pq-device-resident-source-contract-test:
	python3 $(PQ_DEVICE_RESIDENT_SOURCE_CONTRACT_TEST)

pq-request-ring-source-contract-test:
	python3 $(PQ_REQUEST_RING_SOURCE_CONTRACT_TEST)

pq-finite-scheduler-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 \
			$(PQ_FINITE_SCHEDULER_TEST) -o "$$tmp"; \
		"$$tmp"

durable-csv-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 $(DURABLE_CSV_TEST) \
			-o "$$tmp"; \
		"$$tmp"

rerank-vector-store-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra -fopenmp $(HOST_CPPFLAGS) -std=c++17 $(RERANK_VECTOR_STORE_TEST) \
			data/dataloader.cpp index/graph_file_format.cpp index/range_posting_index.cpp -pthread -o "$$tmp"; \
		"$$tmp"

rerank-vector-storage-config-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 \
			$(RERANK_VECTOR_STORAGE_CONFIG_TEST) app/config_loader.cpp -o "$$tmp"; \
		"$$tmp"

rabitq-multiquery-source-contract-test:
	python3 $(RABITQ_MULTIQUERY_SOURCE_CONTRACT_TEST)

.PHONY: rabitq-posting-schedule-config-test
.PHONY: posting-completion-cuda-test
posting-completion-cuda-test:
	@tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(NVCC) $(PROJECT_CPPFLAGS) --std=c++17 -O2 -arch=$(CUDA_CAPABILITY) -Wno-deprecated-gpu-targets \
		-Xcompiler=-pthread ../tests/native/posting_completion_cuda_test.cu -o "$$tmp" && "$$tmp"

.PHONY: posting-completion-test posting-batch-pipeline-test
posting-batch-pipeline-test:
	@tmp=$$(mktemp /tmp/posting-batch-pipeline-test.XXXXXX); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra -std=c++17 -pthread ../tests/native/posting_batch_pipeline_test.cpp -o "$$tmp"; "$$tmp"

posting-completion-test:
	@tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra -std=c++17 -pthread ../tests/native/posting_completion_test.cpp -o "$$tmp" && "$$tmp"

.PHONY: posting-segment-test
posting-segment-test:
	@tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra -std=c++17 ../tests/native/posting_segment_test.cpp -o "$$tmp" && "$$tmp"

rabitq-posting-schedule-config-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 \
			../tests/native/rabitq_posting_schedule_config_test.cpp app/config_loader.cpp -o "$$tmp"; \
		"$$tmp"

.PHONY: cpu-rerank-schedule-test
cpu-rerank-schedule-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(HOST_CPPFLAGS) -O2 -Wall -Wextra -std=c++17 -pthread \
			../tests/native/cpu_rerank_schedule_test.cpp app/config_loader.cpp -o "$$tmp"; \
		"$$tmp"

.PHONY: rabitq-candidate-merge-test
rabitq-candidate-merge-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra -std=c++17 ../tests/native/rabitq_candidate_merge_test.cpp -o "$$tmp"; \
		"$$tmp"

rabitq-lowfreq-merge-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 $(RABITQ_LOWFREQ_MERGE_TEST) \
			app/config_loader.cpp -o "$$tmp"; \
		"$$tmp"

posting-shortlist-heap-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 \
			$(POSTING_SHORTLIST_HEAP_TEST) -o "$$tmp"; \
		"$$tmp"

rabitq-finite-config-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 \
		$(RABITQ_FINITE_CONFIG_TEST) app/config_loader.cpp -o "$$tmp"; \
		"$$tmp"

rabitq-finite-source-contract-test:
	python3 $(RABITQ_FINITE_SOURCE_CONTRACT_TEST)

rabitq-mailbox-rtt-source-contract-test:
	python3 $(RABITQ_MAILBOX_RTT_SOURCE_CONTRACT_TEST)

rabitq-profile-source-contract-test:
	python3 $(RABITQ_PROFILE_SOURCE_CONTRACT_TEST)

rabitq-float-gpu-rerank-source-contract-test:
	python3 $(RABITQ_FLOAT_GPU_RERANK_SOURCE_CONTRACT_TEST)

range-posting-index-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 \
			$(RANGE_POSTING_INDEX_TEST) index/range_posting_index.cpp -o "$$tmp"; \
		"$$tmp"

range-posting-generator-test:
	cd $(ROOT_DIR) && python3 -m $(RANGE_POSTING_GENERATOR_TEST)

range-hierarchy-generator-test:
	cd $(ROOT_DIR) && python3 -m $(RANGE_HIERARCHY_GENERATOR_TEST)

graph-builder-multiworker-source-contract-test:
	python3 $(GRAPH_BUILDER_MULTIWORKER_SOURCE_CONTRACT_TEST)

range-posting-config-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 \
			$(RANGE_POSTING_CONFIG_TEST) app/config_loader.cpp -o "$$tmp"; \
		"$$tmp"

range-access-planner-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 \
			$(RANGE_ACCESS_PLANNER_TEST) $(DATALOADER_TEST_SOURCES) \
			-pthread -o "$$tmp"; \
		"$$tmp"

range-posting-source-contract-test:
	python3 $(RANGE_POSTING_SOURCE_CONTRACT_TEST)

.PHONY: rabitq-cpu-worker-pool-test
rabitq-cpu-worker-pool-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 \
			../tests/native/rabitq_cpu_worker_pool_test.cpp -pthread -o "$$tmp"; \
		timeout 45s "$$tmp"

rabitq-cpu-worker-budget-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 \
			$(RABITQ_CPU_WORKER_BUDGET_TEST) -pthread -o "$$tmp"; \
		"$$tmp"

runtime-route-selection-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra -fopenmp $(HOST_CPPFLAGS) -std=c++17 \
			$(GRAPH_ROUTE_SELECTION_TEST) $(DATALOADER_TEST_SOURCES) \
			-pthread -o "$$tmp"; \
		"$$tmp"

candidate-stream-source-contract-test:
	python3 $(CANDIDATE_STREAM_SOURCE_CONTRACT_TEST)

full-gpu-optimization-source-contract-test:
	python3 $(FULL_GPU_OPTIMIZATION_SOURCE_CONTRACT_TEST)

range-posting-merge-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 \
			$(RANGE_POSTING_MERGE_TEST) -o "$$tmp"; \
		"$$tmp"

operator-fuzzy-rewrite-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 $(OPERATOR_FUZZY_REWRITE_TEST) \
			data/dataloader.cpp app/config_loader.cpp index/range_posting_index.cpp \
			index/graph_file_format.cpp -pthread -o "$$tmp"; \
		"$$tmp"

.PHONY: query-preparation-parallel-test
query-preparation-parallel-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 ../tests/native/query_preparation_parallel_test.cpp \
			$(DATALOADER_TEST_SOURCES) -pthread -o "$$tmp"; \
		"$$tmp"

or-route-planner-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 $(OR_ROUTE_PLANNER_TEST) \
			$(DATALOADER_TEST_SOURCES) -pthread -o "$$tmp"; \
		"$$tmp"

run:
	LD_LIBRARY_PATH=$(CUDA_TOOLKIT_LIB_PATH):$(DEPENDENCY_PREFIX)/lib:$(LIB_PATH):$(LD_LIBRARY_PATH) ./$(TARGET)

clean:
	rm -f $(TARGET)
	rm -rf $(OBJ_DIR) $(TEST_BIN_DIR)

.PHONY: query-setup-timing-test cpu-rerank-profile-test fp16-l2-test performance-trace-test
performance-trace-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -I$(DEPENDENCY_PREFIX)/include -I$(CUDA_TOOLKIT_INCLUDE) -std=c++17 \
			../tests/native/performance_trace_test.cpp -pthread -ldl -o "$$tmp"; "$$tmp"

fp16-l2-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O3 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 \
			../tests/native/fp16_l2_test.cpp -o "$$tmp"; "$$tmp"

cpu-rerank-profile-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 \
			../tests/native/cpu_rerank_profile_test.cpp -pthread -o "$$tmp"; \
		"$$tmp"

query-setup-timing-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 \
			../tests/native/query_setup_timing_test.cpp -o "$$tmp"; \
		"$$tmp"

.PHONY: gpu-predicate-index-test
gpu-predicate-index-test:
	@tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -std=c++17 $(HOST_CPPFLAGS) -I. ../tests/native/gpu_predicate_index_test.cpp -l:libcrypto.so.3 -o "$$tmp" && "$$tmp"

.PHONY: gpu-predicate-cuda-test-build
gpu-predicate-cuda-test-build:
	@mkdir -p $(TEST_BIN_DIR)
	$(NVCC) $(PROJECT_CPPFLAGS) --std=c++17 -O2 -arch=$(CUDA_CAPABILITY) -I. -DGO_FANN_GPU_PREDICATE_CSR_THRESHOLD=$(GPU_PREDICATE_CSR_THRESHOLD) ../tests/native/gpu_predicate_cuda_test.cu -o $(TEST_BIN_DIR)/gpu_predicate_cuda_test

.PHONY: posting-predicate-plan-test posting-predicate-source-contract-test posting-predicate-cuda-test-build
posting-predicate-plan-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra -std=c++17 $(HOST_CPPFLAGS) \
			../tests/native/posting_predicate_plan_test.cpp -o "$$tmp"; "$$tmp"

posting-predicate-source-contract-test:
	python3 ../tests/native/posting_predicate_source_contract_test.py
	python3 ../tests/native/pq_posting_predicate_source_contract_test.py

posting-predicate-cuda-test-build:
	@mkdir -p $(TEST_BIN_DIR)
	$(NVCC) $(PROJECT_CPPFLAGS) --std=c++17 -O2 -arch=$(CUDA_CAPABILITY) -I. ../tests/native/posting_predicate_cuda_test.cu -o $(TEST_BIN_DIR)/posting_predicate_cuda_test

.PHONY: pq-resident-posting-test pq-resident-posting-cuda-test-build
pq-resident-posting-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra -std=c++17 $(HOST_CPPFLAGS) \
			../tests/native/pq_resident_posting_test.cpp -o "$$tmp"; "$$tmp"

pq-resident-posting-cuda-test-build:
	@mkdir -p $(TEST_BIN_DIR)
	$(NVCC) $(PROJECT_CPPFLAGS) --std=c++17 -O2 -arch=$(CUDA_CAPABILITY) -I. ../tests/native/pq_resident_posting_cuda_test.cu -o $(TEST_BIN_DIR)/pq_resident_posting_cuda_test

.PHONY: gpu-predicate-parallel-test-build
gpu-predicate-parallel-test-build:
	@mkdir -p $(TEST_BIN_DIR)
	$(NVCC) $(PROJECT_CPPFLAGS) --std=c++17 -O2 -arch=$(CUDA_CAPABILITY) -I. -DGO_FANN_GPU_PREDICATE_CSR_THRESHOLD=$(GPU_PREDICATE_CSR_THRESHOLD) -DGO_FANN_GPU_PREDICATE_PARALLEL=0 ../tests/native/gpu_predicate_parallel_test.cu -o $(TEST_BIN_DIR)/gpu_predicate_parallel_scalar_test
	@mkdir -p $(TEST_BIN_DIR)
	$(NVCC) $(PROJECT_CPPFLAGS) --std=c++17 -O2 -arch=$(CUDA_CAPABILITY) -I. -DGO_FANN_GPU_PREDICATE_CSR_THRESHOLD=$(GPU_PREDICATE_CSR_THRESHOLD) -DGO_FANN_GPU_PREDICATE_PARALLEL=1 ../tests/native/gpu_predicate_parallel_test.cu -o $(TEST_BIN_DIR)/gpu_predicate_parallel_test

.PHONY: gpu-predicate-csr-lookup-test gpu-predicate-csr-lookup-cuda-test-build
gpu-predicate-csr-lookup-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra -std=c++17 $(HOST_CPPFLAGS) -I. -DGO_FANN_GPU_PREDICATE_CSR_THRESHOLD=$(GPU_PREDICATE_CSR_THRESHOLD) ../tests/native/gpu_predicate_csr_lookup_test.cpp -o "$$tmp"; "$$tmp"

# Build only; the coordinator serializes all CUDA execution with idle checks.
gpu-predicate-csr-lookup-cuda-test-build:
	@mkdir -p $(TEST_BIN_DIR)
	$(NVCC) $(PROJECT_CPPFLAGS) --std=c++17 -O2 -arch=$(CUDA_CAPABILITY) -I. -DGO_FANN_GPU_PREDICATE_CSR_THRESHOLD=$(GPU_PREDICATE_CSR_THRESHOLD) -DGO_FANN_GPU_PREDICATE_PARALLEL=1 ../tests/native/gpu_predicate_csr_lookup_test.cu -o $(TEST_BIN_DIR)/gpu_predicate_csr_lookup_cuda_test

.PHONY: gpu-predicate-csr-test
gpu-predicate-csr-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra -std=c++17 $(HOST_CPPFLAGS) -I. -DGO_FANN_GPU_PREDICATE_CSR_THRESHOLD=$(GPU_PREDICATE_CSR_THRESHOLD) ../tests/native/gpu_predicate_csr_test.cpp -o "$$tmp"; "$$tmp"

.PHONY: rabitq-predicate-config-test
rabitq-predicate-config-test:
	@tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -std=c++17 $(HOST_CPPFLAGS) ../tests/native/rabitq_predicate_config_test.cpp app/config_loader.cpp -o "$$tmp" && "$$tmp"

.PHONY: gpu-predicate-metadata-test
gpu-predicate-metadata-test:
	@tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -std=c++17 $(HOST_CPPFLAGS) -I. ../tests/native/gpu_predicate_metadata_test.cpp -l:libcrypto.so.3 -o "$$tmp" && "$$tmp"


.PHONY: gpu-predicate-frontier-sync-test-build rabitq-frontier-sync-source-contract-test
gpu-predicate-frontier-sync-test-build:
	@mkdir -p $(TEST_BIN_DIR)
	$(NVCC) $(PROJECT_CPPFLAGS) --std=c++17 -O3 -arch=$(CUDA_CAPABILITY) -I. ../tests/native/gpu_predicate_frontier_sync_test.cu -o $(TEST_BIN_DIR)/gpu_predicate_frontier_sync_test

rabitq-frontier-sync-source-contract-test:
	python3 ../tests/native/rabitq_frontier_sync_source_contract_test.py

.PHONY: route-predicate-plan-test route-predicate-host-test gpu-predicate-main-source-contract-test
gpu-predicate-main-source-contract-test:
	python3 ../tests/native/gpu_predicate_main_source_contract_test.py

route-predicate-plan-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra -std=c++17 $(HOST_CPPFLAGS) \
			../tests/native/route_predicate_plan_test.cpp -o "$$tmp"; "$$tmp"

route-predicate-host-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra -std=c++17 $(HOST_CPPFLAGS) \
			-ffunction-sections -fdata-sections ../tests/native/route_predicate_host_test.cpp $(DATALOADER_TEST_SOURCES) \
			-Wl,--gc-sections -pthread -o "$$tmp"; "$$tmp"

.PHONY: predicate-defaults-test
predicate-defaults-test:
	@set -e; tmp=$$(mktemp /tmp/predicate-defaults-test.XXXXXX); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra -std=c++17 -I. ../tests/native/predicate_defaults_test.cpp -o "$$tmp"; "$$tmp"

.PHONY: navigation-score-policy-test
navigation-score-policy-test:
	@set -e; tmp=$$(mktemp /tmp/navigation-score-policy-test.XXXXXX); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra -std=c++17 -I. ../tests/native/navigation_score_policy_test.cpp -o "$$tmp"; "$$tmp"

.PHONY: navigation-score-source-contract-test
navigation-score-source-contract-test:
	python3 ../tests/native/navigation_score_source_contract_test.py

.PHONY: pq-gpu-predicate-test-build
pq-gpu-predicate-test-build:
	@mkdir -p $(TEST_BIN_DIR)
	$(NVCC) $(PROJECT_CPPFLAGS) --std=c++17 -O2 -arch=$(CUDA_CAPABILITY) -I. -DGO_FANN_GPU_PREDICATE_PARALLEL=$(GPU_PREDICATE_PARALLEL) -DGO_FANN_GPU_PREDICATE_CSR_THRESHOLD=$(GPU_PREDICATE_CSR_THRESHOLD) ../tests/native/pq_gpu_predicate_test.cu -o $(TEST_BIN_DIR)/pq_gpu_predicate_test

.PHONY: route-union-test
route-union-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra $(HOST_CPPFLAGS) -std=c++17 ../tests/native/route_union_test.cpp \
			$(DATALOADER_TEST_SOURCES) -pthread -o "$$tmp"; \
		"$$tmp"

.PHONY: exact-distance-memo-test pq-posting-streaming-test
exact-distance-memo-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra -std=c++17 -pthread ../tests/native/exact_distance_memo_test.cpp -o "$$tmp"; "$$tmp"

pq-posting-streaming-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra -std=c++17 -pthread ../tests/native/pq_posting_streaming_test.cpp -o "$$tmp"; "$$tmp"

.PHONY: hybrid-clause-table-test
hybrid-clause-table-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -O2 -Wall -Wextra -std=c++17 -I. ../tests/native/hybrid_clause_table_test.cpp -o "$$tmp"; "$$tmp"

.PHONY: incremental-rerank-test
incremental-rerank-test:
	@set -e; tmp=$$(mktemp); trap 'rm -f "$$tmp"' EXIT; \
		$(HOST_CXX) $(PROJECT_CPPFLAGS) -std=c++17 -O2 -Wall -Wextra -Werror -pthread ../tests/native/incremental_rerank_test.cpp -o "$$tmp"; "$$tmp"

.PHONY: predicate-cli-test
predicate-cli-test: exe
	PYTHONDONTWRITEBYTECODE=1 python3 ../tests/native/rabitq_predicate_cli_test.py ./go-fann
