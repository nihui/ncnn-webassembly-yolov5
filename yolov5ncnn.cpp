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

#include <benchmark.h>
#include <simpleocv.h>
#include "yolov5.h"

static int draw_fps(cv::Mat& rgba)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgba.cols - label_size.width;

    cv::rectangle(rgba, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255, 255), -1);

    cv::putText(rgba, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0, 255));

    return 0;
}

static YOLOv5* g_yolov5 = 0;

static void on_image_render(cv::Mat& rgba)
{
    if (!g_yolov5)
    {
        g_yolov5 = new YOLOv5;

        g_yolov5->load();
    }

    std::vector<Object> objects;
    g_yolov5->detect(rgba, objects);

    g_yolov5->draw(rgba, objects);

    draw_fps(rgba);
}

#ifdef __EMSCRIPTEN_PTHREADS__

static const unsigned char* rgba_data = 0;
static int w = 0;
static int h = 0;

static ncnn::Mutex lock;
static ncnn::ConditionVariable condition;

static ncnn::Mutex finish_lock;
static ncnn::ConditionVariable finish_condition;

static void worker()
{
    while (1)
    {
        lock.lock();
        while (rgba_data == 0)
        {
            condition.wait(lock);
        }

        cv::Mat rgba(h, w, CV_8UC4, (void*)rgba_data);

        on_image_render(rgba);

        rgba_data = 0;

        lock.unlock();

        finish_lock.lock();
        finish_condition.signal();
        finish_lock.unlock();
    }
}

#include <thread>
static std::thread t(worker);

extern "C" {

void yolov5_ncnn(unsigned char* _rgba_data, int _w, int _h)
{
    lock.lock();
    while (rgba_data != 0)
    {
        condition.wait(lock);
    }

    rgba_data = _rgba_data;
    w = _w;
    h = _h;

    lock.unlock();

    condition.signal();

    // wait for finished
    finish_lock.lock();
    while (rgba_data != 0)
    {
        finish_condition.wait(finish_lock);
    }
    finish_lock.unlock();
}

}

#else // __EMSCRIPTEN_PTHREADS__

extern "C" {

void yolov5_ncnn(unsigned char* rgba_data, int w, int h)
{
    cv::Mat rgba(h, w, CV_8UC4, (void*)rgba_data);

    on_image_render(rgba);
}

}

#endif // __EMSCRIPTEN_PTHREADS__
