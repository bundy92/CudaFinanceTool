# CUDA Finance Tool Makefile
# Build system for CUDA option pricing and risk analysis

# Compiler and flags
NVCC = nvcc
CC = gcc
CXX = g++

# CUDA architecture (adjust based on your GPU)
CUDA_ARCH = -arch=sm_60

# Compiler flags
NVCC_FLAGS = $(CUDA_ARCH) -O3 -std=c++11 -Xcompiler -Wall,-Wextra
CXX_FLAGS = -O3 -std=c++11 -Wall -Wextra
CC_FLAGS = -O3 -Wall -Wextra

# Directories
SRC_DIR = .
BUILD_DIR = build
BIN_DIR = bin
INCLUDE_DIR = .

# Source files
CUDA_SOURCES = $(wildcard $(SRC_DIR)/*.cu)
CXX_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
C_SOURCES = $(wildcard $(SRC_DIR)/*.c)

# Web interface
WEB_DIR = web_interface
WEB_APP = $(WEB_DIR)/app.py

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)
CXX_OBJECTS = $(CXX_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
C_OBJECTS = $(C_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Executables
MAIN_EXEC = $(BIN_DIR)/cuda_finance_tool
TEST_EXEC = $(BIN_DIR)/test_suite

# Default target
all: directories $(MAIN_EXEC)

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Main executable
$(MAIN_EXEC): $(CUDA_OBJECTS) $(CXX_OBJECTS) $(C_OBJECTS)
	@echo "Linking $@..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $^
	@echo "Build complete: $@"

# Compile CUDA source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "Compiling $<..."
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Compile C++ source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Compile C source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling $<..."
	$(CC) $(CC_FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Test target
test: directories $(TEST_EXEC)
	@echo "Running tests..."
	./$(TEST_EXEC)

# Web interface target
web: directories
	@echo "Starting web interface..."
	cd $(WEB_DIR) && python app.py

# Install web dependencies
install-web:
	@echo "Installing web dependencies..."
	pip install -r requirements.txt

$(TEST_EXEC): $(BUILD_DIR)/test_suite.o $(CUDA_OBJECTS)
	@echo "Linking test executable..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Clean target
clean:
	@echo "Cleaning build files..."
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Install target (optional)
install: all
	@echo "Installing to /usr/local/bin..."
	sudo cp $(MAIN_EXEC) /usr/local/bin/

# Uninstall target
uninstall:
	@echo "Uninstalling..."
	sudo rm -f /usr/local/bin/cuda_finance_tool

# Help target
help:
	@echo "Available targets:"
	@echo "  all        - Build the main executable"
	@echo "  test       - Build and run tests"
	@echo "  web        - Start web interface"
	@echo "  install-web - Install web dependencies"
	@echo "  clean      - Remove build files"
	@echo "  install    - Install to system"
	@echo "  uninstall  - Remove from system"
	@echo "  help       - Show this help"

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version || (echo "CUDA not found. Please install CUDA Toolkit." && exit 1)
	@echo "CUDA installation found."

# Check GPU compatibility
check-gpu:
	@echo "Checking GPU compatibility..."
	@nvidia-smi || (echo "NVIDIA GPU not found or drivers not installed." && exit 1)
	@echo "GPU found and accessible."

# Setup target (run before first build)
setup: check-cuda check-gpu
	@echo "Setup complete. Ready to build."

.PHONY: all directories clean install uninstall help check-cuda check-gpu setup test 