#include "entroyCalibrator.h"
#include "pre.h"
#include "config.h"
#include <cuda_runtime_api.h>
#include "NvOnnxParser.h"
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <iterator>
#include <iostream>
#include "NvInfer.h" 
#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }
#endif

namespace nvinfer1
{

    int8EntroyCalibrator::int8EntroyCalibrator(const int &bacthSize, const std::string &imgPath,
        const std::string &calibTablePath):batchSize(bacthSize),calibTablePath(calibTablePath),imageIndex(0){ 

        int inputChannel = centernet::INPUT_CLS_SIZE;
        int inputH = centernet::INPUT_H;
        int inputW = centernet::INPUT_W;
        inputCount = bacthSize * inputChannel * inputH * inputW;

        std::fstream f(imgPath);
        if(f.is_open()){
            std::string temp;
            while (std::getline(f,temp)) imgPaths.push_back(temp);

        }
        batchData = new float[inputCount];

        CUDA_CHECK(cudaMalloc(&deviceInput, inputCount * sizeof(float)));
    }

    int8EntroyCalibrator::~int8EntroyCalibrator() {
        CUDA_CHECK(cudaFree(deviceInput));
        if(batchData)
            delete[] batchData;
    }

    bool int8EntroyCalibrator::getBatch(void **bindings, const char **names, int nbBindings){
        if (imageIndex + batchSize > int(imgPaths.size()))
            return false;
        // load batch
        float* ptr = batchData;
        for (size_t j = imageIndex; j < imageIndex + batchSize; ++j)
        {
            auto img = cv::imread(imgPaths[j]);
            auto inputData = prepareImage(img);
            if (inputData.size() != inputCount / batchSize) // batch size = 1, inputData.size() = inputCount, Now bs = 10
            {
                std::cout << "InputSize error." << std::endl;
                return false;
            }
            assert(inputData.size() == inputCount / batchSize);
            memcpy(ptr,inputData.data(),inputData.size()*sizeof(float));

            ptr += inputData.size();
            std::cout << "load image " << imgPaths[j] << "  " << (j+1)*100./imgPaths.size() << "%" << std::endl;
        }
        imageIndex += batchSize;
        CUDA_CHECK(cudaMemcpy(deviceInput, batchData, inputCount * sizeof(float), cudaMemcpyHostToDevice));
        bindings[0] = deviceInput;
        return true;
    }
    const void* int8EntroyCalibrator::readCalibrationCache(std::size_t &length)
    {
        calibrationCache.clear();
        std::ifstream input(calibTablePath, std::ios::binary);
        input >> std::noskipws;
        if (readCache && input.good())
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                    std::back_inserter(calibrationCache));

        length = calibrationCache.size();
        return length ? &calibrationCache[0] : nullptr;
    }

    void int8EntroyCalibrator::writeCalibrationCache(const void *cache, std::size_t length)
    {
        std::ofstream output(calibTablePath, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

}
