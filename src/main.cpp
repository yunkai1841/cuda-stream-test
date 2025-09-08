#include <cuda_runtime.h>
#include <gflags/gflags.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>
#include <iomanip> // std::setprecision

#include "cuda_utils.h"
#include "matrixmul.cuh"

// コマンドライン引数の定義
DEFINE_int32(matrix_size, 1024, "Matrix size (N x N)");
DEFINE_int32(num_async, 4, "Number of asynchronous matrix multiplications");
DEFINE_string(kernel_type, "basic", "Kernel type: basic");
DEFINE_int32(use_streams, 1, "Use CUDA streams for asynchronous execution");
DEFINE_string(performance_report, "", "Performance report output file (empty: do not save)");
DEFINE_int32(stream_limit, -1, "Maximum amount of streams");
DEFINE_int32(nums_in_stream, 1, "Maximum amount of kernels pushed into a stream every time(set minus to disable spliting)");
DEFINE_int32(small_num, 0, "Amount of small matrixs(half * half)");
using namespace cuda_utils;

// 行列の初期化
void initializeMatrix(float* matrix, int N, float value = 1.0f) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = value;
    }
}

// ラッパー関数: std::stringをconst char*に変換してlauncher.cuの関数を呼び出し
void launchMatrixMulKernelWrapper(const std::string& kernel_type, float* d_C, const float* d_A,
                                  const float* d_B, int N, cudaStream_t stream) {
    launchMatrixMulKernel(kernel_type.c_str(), d_C, d_A, d_B, N, stream);
}

int main(int argc, char** argv) {

    // --use_streams false のようなスペース区切り指定を検出し、エラー終了
    for (int i = 1; i < argc - 1; ++i) {
        if ((std::string(argv[i]) == "--use_streams" || std::string(argv[i]) == "-use_streams") &&
            (std::string(argv[i+1]) == "false" || std::string(argv[i+1]) == "0" || std::string(argv[i+1]) == "true" || std::string(argv[i+1]) == "1")) {
            std::cerr << "[ERROR] --use_streams の指定は --use_streams=false のようにイコール区切りで指定してください。" << std::endl;
            return 1;
        }
    }
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const int N = FLAGS_matrix_size;
    const int num_async = FLAGS_num_async;
    const std::string kernel_type = FLAGS_kernel_type;
    const int use_streams = FLAGS_use_streams;
   const int nums_in_stream = FLAGS_nums_in_stream;
   const int small_num = FLAGS_small_num;
   const int stream_limit = (FLAGS_stream_limit <= 0 || FLAGS_stream_limit > FLAGS_num_async) ? FLAGS_num_async : FLAGS_stream_limit;
   const int M = N / 2;
    std::cout << "Matrix Multiplication Configuration:" << std::endl;
    std::cout << "Matrix Multiplication Configuration:" << std::endl;
    std::cout << "  Matrix size: " << N << " x " << N << std::endl;
        std::cout << "  Matrix size(small): " << M << " x " << M << std::endl;
        
    std::cout << "  Number of async operations: " << num_async << std::endl;
    std::cout << "  Small Matrix Percentage: " << (small_num * 10000 / num_async) / 100.0 << "%" <<std::endl;
    std::cout << "  Kernel type: " << kernel_type << std::endl;
    std::cout << "  Use streams: " << (use_streams ? "Yes" : "No") << std::endl;
	std::cout << "  Stream limit: " << stream_limit << std::endl;
    std::cout << "  Kernels in stream: " << nums_in_stream<< std::endl;
    // メモリサイズの計算
    const size_t matrix_bytes = N * N * sizeof(float);
    
    const size_t matrix_bytes_small = M * M * sizeof(float);

    // ホストメモリの確保
    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_A_s(M * M);
    std::vector<float> h_B_s(M * M);
    std::vector<std::vector<float>> h_C_results(num_async, std::vector<float>(N * N));
    std::vector<std::vector<float>> h_C_results_s(num_async, std::vector<float>(M * M));

    // 行列の初期化
    initializeMatrix(h_A.data(), N, 1.7f);
    initializeMatrix(h_B.data(), N, 1.1f);

    initializeMatrix(h_A_s.data(), M, 1.7f);
    initializeMatrix(h_B_s.data(), M, 1.1f);
    // デバイスメモリの確保（スマートポインタとRAII）
    auto d_A = std::make_unique<CudaMemory>(matrix_bytes);
    auto d_B = std::make_unique<CudaMemory>(matrix_bytes);
    std::vector<std::unique_ptr<CudaMemory>> d_C_array;
    
    auto d_A_s = std::make_unique<CudaMemory>(matrix_bytes_small);
    auto d_B_s = std::make_unique<CudaMemory>(matrix_bytes_small);
    std::vector<std::unique_ptr<CudaMemory>> d_C_array_s;

    for (int i = 0; i < num_async; i++) {
        d_C_array.push_back(std::make_unique<CudaMemory>(matrix_bytes));
    }
    for (int i = 0; i < num_async; i++) {
        d_C_array_s.push_back(std::make_unique<CudaMemory>(matrix_bytes_small));
    }
	
	
	
   //make random sequence for different sizes of tasks
    int* sequence = (int*)malloc(sizeof(int) * num_async);
    memset(sequence, 0, sizeof(sequence));
    for(int i = 0; i < small_num; i++)sequence[i] = 1;
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::shuffle(sequence, sequence + num_async, gen);
    int tempt = 0;
    for(int i = 0; i < num_async; i++){
    	std::cout << sequence[i];
    	tempt += sequence[i];
    	}
    std::cout << '\n' << tempt;	
    
    
    // CUDAストリームの作成（スマートポインタとRAII）
    std::vector<std::unique_ptr<CudaStream>> streams;
    if (use_streams) {
        for (int i = 0; i < num_async; i++) {
            streams.push_back(std::make_unique<CudaStream>());
        }
    }

    // ホストからデバイスへのデータ転送
    cudaMemcpy(d_A->get(), h_A.data(), matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B->get(), h_B.data(), matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_s->get(), h_A_s.data(), matrix_bytes_small, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_s->get(), h_B_s.data(), matrix_bytes_small, cudaMemcpyHostToDevice);

    // CUDAタイマーの作成
    CudaTimer total_timer;
    std::vector<CudaTimer> kernel_timers(num_async);

    // 実行時間の測定開始（CPU時間）
    auto start = std::chrono::high_resolution_clock::now();
    
    // 全体のCUDA実行時間測定開始
    total_timer.start();
    int num_str = 0;

    int temp = 0;
    // 非同期でmatrix multiplicationを実行（各カーネルの時間も計測）
    if(nums_in_stream <= 1){
    num_str = (nums_in_stream < 0) ? -1 * (num_async / nums_in_stream) : (num_async / nums_in_stream);
    
	    for (int i = 0; i < num_async; i++) {
		cudaStream_t stream = use_streams ? streams[((nums_in_stream < 0) ? -1 * (i / nums_in_stream) : (i / nums_in_stream)) % stream_limit]->get() : 0;
		if(sequence[i])
			launchMatrixMulKernelWrapper(kernel_type, d_C_array_s[i]->get(), d_A_s->get(), d_B_s->get(), M,
		                             stream);
		else                             
			launchMatrixMulKernelWrapper(kernel_type, d_C_array[i]->get(), d_A->get(), d_B->get(), N,
		                             stream);
	    }
    }
    else{

	    for (int i =  0; (double)i  <  num_async * 0.90; i++) {
		cudaStream_t stream = use_streams ? streams[num_str % stream_limit]->get() : 0;
		if(sequence[i])
			launchMatrixMulKernelWrapper(kernel_type, d_C_array_s[i]->get(), d_A_s->get(), d_B_s->get(), M,
		                             stream);
		else                             
			launchMatrixMulKernelWrapper(kernel_type, d_C_array[i]->get(), d_A->get(), d_B->get(), N,
		                             stream);
		temp++;  
		if(temp == nums_in_stream)   
		{
			num_str++; 
			temp = 0;
		}                       

	    }

	    for (int i =  (int)(num_async * 0.90); i < num_async; i++) {
		cudaStream_t stream = use_streams ? streams[num_str % stream_limit]->get() : 0;
		if(sequence[i])
			launchMatrixMulKernelWrapper(kernel_type, d_C_array_s[i]->get(), d_A_s->get(), d_B_s->get(), M,
		                             stream);
		else                             
			launchMatrixMulKernelWrapper(kernel_type, d_C_array[i]->get(), d_A->get(), d_B->get(), N,
		                             stream);
		num_str++;                            

	    }
     }

	int min_ker = (stream_limit > num_str) ? num_str : stream_limit;
    // すべてのカーネルの完了を待機
    if (use_streams) {
        for (int i = 0; i < min_ker; i++) {
            streams[i]->synchronize();
        }
    } else {
        cudaDeviceSynchronize();
    }
    
    // 全体のCUDA実行時間測定終了
    total_timer.stop();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start); // double型のms

    // 結果をホストにコピ
    
    if(nums_in_stream <= 1){
    //last 10% of the kernel remain unchanged
    	for (int i = 0; i < num_async; i++) {
		cudaStream_t stream_cpy = use_streams ? streams[((nums_in_stream < 0) ? -1 * (i / nums_in_stream) : (i / nums_in_stream)) % stream_limit]->get() : 0;
		if(sequence[i])
			cudaMemcpyAsync(h_C_results_s[i].data(), d_C_array_s[i]->get(), matrix_bytes_small,
		                cudaMemcpyDeviceToHost, stream_cpy);
		else
			cudaMemcpyAsync(h_C_results[i].data(), d_C_array[i]->get(), matrix_bytes,
		                cudaMemcpyDeviceToHost, stream_cpy);                           
	    }
    }
    else{
    //split the last 10% kernels, push into stream one at a time
	    num_str = 0;
	    temp = 0;
	    for (int i =  0; (double)i  <  num_async * 0.90; i++) {
		cudaStream_t stream_cpy = use_streams ? streams[num_str % stream_limit]->get() : 0;
		if(sequence[i])
			cudaMemcpyAsync(h_C_results_s[i].data(), d_C_array_s[i]->get(), matrix_bytes_small,
		                cudaMemcpyDeviceToHost, stream_cpy);
		else
			cudaMemcpyAsync(h_C_results[i].data(), d_C_array[i]->get(), matrix_bytes,
		                cudaMemcpyDeviceToHost, stream_cpy);   
		temp++;  
		if(temp == nums_in_stream)   
		{
			num_str++; 
			temp = 0;
		}                       

	    }

	    for (int i =  (int)(num_async * 0.90); i < num_async; i++) {
		cudaStream_t stream_cpy = use_streams ? streams[num_str % stream_limit]->get() : 0;
		if(sequence[i])
			cudaMemcpyAsync(h_C_results_s[i].data(), d_C_array_s[i]->get(), matrix_bytes_small,
		                cudaMemcpyDeviceToHost, stream_cpy);
		else
			cudaMemcpyAsync(h_C_results[i].data(), d_C_array[i]->get(), matrix_bytes,
		                cudaMemcpyDeviceToHost, stream_cpy);   
		num_str++;                            

	    }
    }
    /* すべての転送完了を待機*/
    min_ker = (stream_limit > num_str) ? num_str : stream_limit;
    if (use_streams) {
        for (int i = 0; i < min_ker ; i++) {
            streams[i]->synchronize();
        }
    } else {
        cudaDeviceSynchronize();
    }

    // 結果の検証（最初の要素のみ）
    std::cout << "\nResults verification:" << std::endl;
    for (int i = 0; i < num_async; i+=16) {
        std::cout << "  Operation " << i << ": C[0][0] = " << (sequence[i] ? h_C_results_s[i][0] : h_C_results[i][0]) << std::endl;
    }

    // パフォーマンス結果の表示
    std::cout << "\nPerformance Results:" << std::endl;
    std::cout << std::fixed << std::setprecision(2); // 小数点以下2桁で表示
    std::cout << "  Total CPU execution time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Total CUDA execution time: " << total_timer.elapsedMilliseconds() << " ms" << std::endl;
    std::cout << "  Average CPU time per operation: " << duration.count() / num_async << " ms" << std::endl;
    
    // 各カーネルの実行時間を表示
    //std::cout << "\nIndividual Kernel Execution Times:" << std::endl;
    float total_kernel_time = 0.0f;
    for (int i = 0; i < num_async; i++) {
        float kernel_time = kernel_timers[i].elapsedMilliseconds();
        total_kernel_time += kernel_time;
        //std::cout << "  Kernel " << i << ": " << kernel_time << " ms" << std::endl;
    }
    std::cout << "  Sum of individual kernel times: " << total_kernel_time << " ms" << std::endl;
    std::cout << "  Average kernel time: " << total_kernel_time / num_async << " ms" << std::endl;

    // 計算スループットの計算（GFLOPS）
    double flops = 2.0 * N * N * N * (num_async - small_num) + 2.0 * M * M * M * (small_num) ;        // 各行列乗算で2*N^3 FLOPs
    double gflops_cpu = flops / (duration.count() * 1e6);  // CPU時間ベースのGFLOPS
    double gflops_cuda = flops / (total_timer.elapsedMilliseconds() * 1e6);  // CUDA時間ベースのGFLOPS
    std::cout << "  Throughput (CPU time): " << gflops_cpu << " GFLOPS" << std::endl;
    std::cout << "  Throughput (CUDA time): " << gflops_cuda << " GFLOPS" << std::endl;

    // パフォーマンスレポートをファイルに保存（引数が指定された場合のみ）
    if (!FLAGS_performance_report.empty()) {
        std::ofstream ofs(FLAGS_performance_report);
        ofs << "Performance Results:" << std::endl;
        ofs << std::fixed << std::setprecision(2);
        //ofs << "  Total CPU execution time: " << duration.count() << " ms" << std::endl;
        ofs << "  Total CUDA execution time: " << total_timer.elapsedMilliseconds() << " ms" << std::endl;
        //ofs << "  Average CPU time per operation: " << duration.count() / num_async << " ms" << std::endl;
        ofs << "  Average kernel time: " << total_kernel_time / num_async << " ms" << std::endl;
        //ofs << "  Throughput (CPU time): " << gflops_cpu << " GFLOPS" << std::endl;
        ofs << "  Throughput (CUDA time): " << gflops_cuda << " GFLOPS" << std::endl;
        
        ofs << "\nIndividual Kernel Times:" << std::endl;
        for (int i = 0; i < num_async; i++) {
            //ofs << "  Kernel " << i << ": " << kernel_timers[i].elapsedMilliseconds() << " ms" << std::endl;
        }
    }

    // クリーンアップ（スマートポインタが自動的に行うため、手動は不要）
    // streams と d_C_array は自動的にデストラクタが呼ばれる

    std::cout << "\nMatrix multiplication completed successfully!" << std::endl;
    return 0;
}
