/*************************************************************************
	> File Name: pre.cpp
	> Author: leecheer
	> Mail: liqian@1000video.com.cn
	> Function: 
	> Created Time: 2020年11月04日 星期三 17时01分52秒
 ************************************************************************/
#include "pre.h"
#include "config.h"
#include <sstream>

std::vector<float> prepareImage(cv::Mat& img) {

    int channel = centernet::INPUT_CLS_SIZE;
    int input_w = centernet::INPUT_W;
    int input_h = centernet::INPUT_H;
    // float scale = cv::min(float(input_w)/img.cols,float(input_h)/img.rows);
    auto scaleSize = cv::Size(input_w, input_h);

    cv::Mat resized, ft32;
    cv::resize(img, resized, scaleSize, cv::INTER_NEAREST);
    resized.convertTo(ft32, CV_32FC3);

    //HWC TO CHW
    std::vector<cv::Mat> input_channels(channel);
    cv::split(ft32, input_channels);

    // normalize
    std::vector<float> result(input_h * input_w * channel);
    auto data = result.data();
    int channelLength = input_h * input_w;
    for (int i = 0; i < channel; ++i) {
        memcpy(data, input_channels[i].data, channelLength * sizeof(float));
        data += channelLength;
    }
    return result;
} 

