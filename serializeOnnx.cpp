#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "argsParser.h"
#include "logger.h"
#include "common.h"
#include "entroyCalibrator.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

const std::string gSampleName = "TensorRT.sample_onnx_centernet";

using namespace nvinfer1;


static const int INPUT_H = 256;
static const int INPUT_W = 256;
static const int INPUT_CLS_SIZE = 3;
static const int OUTPUT_CLS_SIZE = 2;
static const int OUTPUT_SIZE_H = 64;
static const int OUTPUT_SIZE_W = 64;
static const int batchsize = 10;
static const int detsLength = 7;

const std::string CLASSES[OUTPUT_CLS_SIZE]{"carringbag", "umbrella"};

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME0 = "wh";
const char* OUTPUT_BLOB_NAME1 = "reg";
const char* OUTPUT_BLOB_NAME2 = "hm";
const char* OUTPUT_BLOB_NAME3 = "h_max";

const char* gNetworkName{nullptr};

samplesCommon::Args gArgs;


void mat_to_array(cv::Mat in,float *out){
    int arrayCount = 0;
    int channel = in.channels();
    int height = in.rows;
    int width = in.cols;
    for (int c = 0; c < channel; c++) {
        for (int i=0; i < height; i++) {
             for (int j =0; j < width; j++,arrayCount++){
                out[arrayCount] = in.at<cv::Vec3f>(i,j)[c];
                // printf("第%d行，第%d列，第%d通道的值：%d\n", i+1, j+1, c+1, int(out[arrayCount]));
            }
        }
    }
}


// output the wh, reg, hm, h_max 
void writefeature(float*wh, float* reg, float* hm, float* h_max, float* data){ 
    int len = OUTPUT_SIZE_W * OUTPUT_SIZE_H * OUTPUT_CLS_SIZE;
    float *pWh = wh; 
    float *pReg = reg; 
    float *pHm = hm; 
    float *pH_max = h_max; 
    float *pData = data;

    fstream fs;

    fs.open("output.txt", ios::out);

    // printf("wh:\n");
    fs.write("wh:\n", 4);
    for (int i = 0; i < len; i++) { 
        fs << *pWh++;
        fs.write("\n",1);
    }

    // reg output:
    fs.write("reg:\n", 5);
    for (int i = 0; i < len; i++) { 
        fs << *pReg++;
        fs.write("\n",1);
    }
    fs.write("\n",1);

    // hm output: 
    fs.write("hm:\n", 4);
    for (int i = 0; i < len; i++) { 
        fs << *pHm++;
        fs.write("\n",1);
    }
    fs.write("\n",1);

    // h_max output:
    fs.write("h_max:\n", 7);
    for (int i = 0; i < len; i++) { 
        // printf("%f", *p++);
        fs << *pH_max++;
        fs.write("\n",1);
    }
    fs.write("\n",1);

    fs.write("data:\n", 6);
    for (int i = 0; i < 3*INPUT_H*INPUT_W; i++) {
        fs << *pData++;
        fs.write("\n", 1);
    }

    // close the fileName
    fs.close();
    gLogInfo << "feature map saved in output.txt.";
}

bool onnxToTRTModel(const std::string& modelFile, // name of the onnx model
                    unsigned int maxBatchSize,    // batch size - NB must be at least as large as the batch we want to run with // IInt8Calibrator* calibrator, // calibrator 
                    IHostMemory*& trtModelStream) // output buffer for the TensorRT model
{
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();
    

    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());



	int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;
    // if ( !parser->parseFromFile( locateFile(modelFile, gArgs.dataDirs).c_str(), static_cast<int>(gLogger.getReportableSeverity()) ) )
    if ( !parser->parseFromFile( locateFile(modelFile, gArgs.dataDirs).c_str(), verbosity) )
    {
        gLogError << "Failure while parsing ONNX file" << std::endl;
        return false;
    }

    const std::string calibFile = "/home/ubuntu/SSD/software/TensorRT-5.1.2.2/data/centernet/carringBag_calibration.txt";
    nvinfer1::int8EntroyCalibrator *calibrator = nullptr; 
    if(calibFile.size() > 0) calibrator = new nvinfer1::int8EntroyCalibrator(maxBatchSize, calibFile, "calib.table");
    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 30);
    // builder->setFp16Mode(false);
    builder->setInt8Mode(true);
    builder->setInt8Calibrator(calibrator);

    
    // samplesCommon::enableDLA(builder, gArgs.useDLACore);
    
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we can destroy the parser
    parser->destroy();

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();
    if (engine)
    {
        // serialize the engine, then close everything down
        trtModelStream = engine->serialize();
		// save the engine
        std::ofstream ofs("int8.engine", std::ios::out | std::ios::binary);
        gLogInfo << "TRT engine saved: /home/ubuntu/SSD/software/TensorRT-5.1.2.2/bin.\n";
        ofs.write((char*)(trtModelStream ->data()), trtModelStream ->size());
        ofs.close();
        engine->destroy();
    }

    if (calibrator)
    {
        delete calibrator;
        calibrator = nullptr;
    }

    // engine->destroy();
    network->destroy();
    builder->destroy();

    return true;
}

/*
void decodeHeatmap(float* wh, float* reg, float* hm, float* h_max, int batchSize, k=100) {
}
*/

void doInference(IExecutionContext& context, float* input, float* wh, float* reg, float* hm, float* h_max, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 5);
    void* buffers[5];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()

    int dataIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
    	whIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME0),
    	regIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME1),
    	hmIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME2),
    	h_maxIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME3);

    const int CLS_IP = 3;
    const int dataSize = batchSize * CLS_IP * INPUT_H * INPUT_W;
    const int CLS_OP = 2;
    const int commonSize = batchSize * CLS_OP * OUTPUT_SIZE_H * OUTPUT_SIZE_W;
    const int whSize = commonSize;
    const int regSize = commonSize;
    const int hmSize = commonSize; 
    const int h_maxSize = commonSize;

    int inputIndex{};
    for (int b = 0; b < engine.getNbBindings(); ++b)
    {
        if (engine.bindingIsInput(b))
            inputIndex = b;
    }

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], dataSize * sizeof(float))); // data

    for (int tempBuffIndex = 1; tempBuffIndex < engine.getNbBindings(); tempBuffIndex++) {
        CHECK(cudaMalloc(&buffers[tempBuffIndex], commonSize * sizeof(float))); // wh, reg, hm, h_max
    }

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, dataSize * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);

    for (int tempBuffIndex = 1; tempBuffIndex < engine.getNbBindings(); tempBuffIndex++) {
		if (tempBuffIndex == whIndex)
        	CHECK(cudaMemcpyAsync(wh, buffers[tempBuffIndex], whSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
		else if (tempBuffIndex == regIndex)
        	CHECK(cudaMemcpyAsync(reg, buffers[tempBuffIndex], regSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
		else if (tempBuffIndex == hmIndex)
        	CHECK(cudaMemcpyAsync(hm, buffers[tempBuffIndex], hmSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
        else if (tempBuffIndex == h_maxIndex) 
            CHECK(cudaMemcpyAsync(h_max, buffers[tempBuffIndex], h_maxSize *  sizeof(float), cudaMemcpyDeviceToHost, stream));
    }

    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[dataIndex]));
    CHECK(cudaFree(buffers[whIndex]));
    CHECK(cudaFree(buffers[regIndex]));
    CHECK(cudaFree(buffers[hmIndex]));
    CHECK(cudaFree(buffers[h_maxIndex]));
}


void printHelpInfo()
{
    std::cout << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)" << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
    std::cout << "--int8          Run in Int8 mode.\n";
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    bool argsOK = samplesCommon::parseArgs(gArgs, argc, argv);
    if (gArgs.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (gArgs.dataDirs.empty())
    {
        gArgs.dataDirs = std::vector<std::string>{"data/samples/centernet/", "data/centernet/"};
    }

    // gLogger.reportTestStart(sampleTest);


    IHostMemory* trtModelStream{nullptr};



    // if (!onnxToTRTModel("batch-1.onnx", 1, trtModelStream))
    if (onnxToTRTModel("centernet_mobilenetv2_10_objdet.onnx", 1, trtModelStream))
        gLogInfo << "convert sucessed.\n";
    else 
        gLogInfo << "can't parse!";
    

}

