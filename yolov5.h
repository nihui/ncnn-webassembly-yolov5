// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef YOLOV5_H
#define YOLOV5_H

#include <net.h>
#include <simpleocv.h>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class YOLOv5
{
public:
    YOLOv5();

    int load();

    int detect(const cv::Mat& rgba, std::vector<Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f);

    int draw(cv::Mat& rgba, const std::vector<Object>& objects);

private:
    ncnn::Net yolov5;
};

#endif // YOLOV5_H
