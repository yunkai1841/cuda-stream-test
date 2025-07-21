#!/bin/bash

# CUDA MPS (Multi-Process Service) testing script
# This script demonstrates concurrent execution with MPS

# Configuration
BIN="../builddir/src/main"
MATRIX_SIZE=1024
NUM_ASYNC=4
KERNEL_TYPE="basic"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}CUDA MPS Testing Script${NC}"
echo "======================================="

# Check if binary exists
if [ ! -f "$BIN" ]; then
    echo -e "${RED}Error: Binary not found at $BIN${NC}"
    echo "Please build the project first:"
    echo "  meson setup builddir"
    echo "  cd builddir && meson compile"
    exit 1
fi

# Function to check if MPS is running
check_mps_status() {
    if pgrep -f "nvidia-cuda-mps-control" > /dev/null; then
        return 0  # MPS is running
    else
        return 1  # MPS is not running
    fi
}

# Function to start MPS
start_mps() {
    echo -e "${YELLOW}Starting MPS daemon...${NC}"
    
    # Set required environment variables
    export CUDA_VISIBLE_DEVICES=0
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
    
    # Create directories if they don't exist
    mkdir -p $CUDA_MPS_PIPE_DIRECTORY
    mkdir -p $CUDA_MPS_LOG_DIRECTORY
    
    # Start MPS daemon
    nvidia-cuda-mps-control -d
    sleep 2
    
    if check_mps_status; then
        echo -e "${GREEN}MPS daemon started successfully${NC}"
        echo "  Pipe directory: $CUDA_MPS_PIPE_DIRECTORY"
        echo "  Log directory: $CUDA_MPS_LOG_DIRECTORY"
        return 0
    else
        echo -e "${RED}Failed to start MPS daemon${NC}"
        return 1
    fi
}

# Function to stop MPS
stop_mps() {
    echo -e "${YELLOW}Stopping MPS daemon...${NC}"
    
    # Set environment variables if they exist
    if [ -n "$CUDA_MPS_PIPE_DIRECTORY" ]; then
        export CUDA_MPS_PIPE_DIRECTORY
    else
        export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    fi
    
    echo quit | nvidia-cuda-mps-control
    sleep 1
    
    if ! check_mps_status; then
        echo -e "${GREEN}MPS daemon stopped successfully${NC}"
        # Clean up directories
        rm -rf /tmp/nvidia-mps 2>/dev/null
        rm -rf /tmp/nvidia-log 2>/dev/null
    else
        echo -e "${RED}Warning: MPS daemon may still be running${NC}"
    fi
}

# Function to run benchmark
run_benchmark() {
    local use_streams=$1
    local use_mps=$2
    local test_name=$3
    
    echo ""
    echo -e "${YELLOW}Running: $test_name${NC}"
    echo "----------------------------------------"
    
    ARGS="--matrix_size=$MATRIX_SIZE --num_async=$NUM_ASYNC --kernel_type=$KERNEL_TYPE"
    ARGS="$ARGS --use_streams=$use_streams --use_mps=$use_mps"
    
    "$BIN" $ARGS
    echo ""
}

# Main test sequence
echo ""
echo "Test Configuration:"
echo "  Matrix Size: $MATRIX_SIZE x $MATRIX_SIZE"
echo "  Number of Async Operations: $NUM_ASYNC"
echo "  Kernel Type: $KERNEL_TYPE"
echo ""

# Test 1: Without MPS, without streams
run_benchmark false false "Test 1: No Streams, No MPS (Sequential)"

# Test 2: Without MPS, with streams
run_benchmark true false "Test 2: With Streams, No MPS"

# Check if MPS is available
echo -e "${YELLOW}Checking MPS availability...${NC}"
"$BIN" --use_mps=true --matrix_size=64 --num_async=1 > /tmp/mps_check.out 2>&1

if grep -q "MPS is not supported" /tmp/mps_check.out; then
    echo -e "${RED}MPS is not supported on this device${NC}"
    echo "Available tests completed."
    exit 0
fi

# Test 3: Start MPS and run tests
if ! check_mps_status; then
    if ! start_mps; then
        echo -e "${RED}Cannot start MPS daemon. Skipping MPS tests.${NC}"
        exit 1
    fi
fi

# Test 4: With MPS, without explicit streams (MPS will handle concurrency)
run_benchmark false true "Test 3: No Streams, With MPS"

# Test 5: With MPS and streams
run_benchmark true true "Test 4: With Streams, With MPS"

# Multi-process test with MPS
echo -e "${YELLOW}Running multi-process test with MPS...${NC}"
echo "Starting 3 concurrent processes..."

ARGS="--matrix_size=$MATRIX_SIZE --num_async=$NUM_ASYNC --kernel_type=$KERNEL_TYPE"
ARGS="$ARGS --use_streams=true --use_mps=true"

# Ensure environment variables are set for child processes
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-log}

# Start multiple processes in background
"$BIN" $ARGS > /tmp/mps_process1.log 2>&1 &
PID1=$!
"$BIN" $ARGS > /tmp/mps_process2.log 2>&1 &
PID2=$!
"$BIN" $ARGS > /tmp/mps_process3.log 2>&1 &
PID3=$!

# Wait for all processes to complete
wait $PID1
wait $PID2
wait $PID3

echo "Multi-process execution completed."
echo ""
echo "Process 1 Results:"
grep "Total CUDA execution time" /tmp/mps_process1.log || echo "No timing data found"
echo ""
echo "Process 2 Results:"
grep "Total CUDA execution time" /tmp/mps_process2.log || echo "No timing data found"
echo ""
echo "Process 3 Results:"
grep "Total CUDA execution time" /tmp/mps_process3.log || echo "No timing data found"

# Cleanup
rm -f /tmp/mps_check.out /tmp/mps_process*.log

# Ask user if they want to stop MPS
echo ""
read -p "Do you want to stop the MPS daemon? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    stop_mps
else
    echo -e "${YELLOW}MPS daemon left running for further testing${NC}"
    echo "To stop manually: echo quit | nvidia-cuda-mps-control"
fi

echo ""
echo -e "${GREEN}MPS testing completed!${NC}"
