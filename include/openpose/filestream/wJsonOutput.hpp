#ifndef OPENPOSE_FILESTREAM_W_JSON_OUTPUT_HPP
#define OPENPOSE_FILESTREAM_W_JSON_OUTPUT_HPP

#include <openpose/core/common.hpp>
#include <openpose/thread/workerConsumer.hpp>
#include <openpose/producer/videoReader.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <opencv2/opencv.hpp>
#include <map>
#include <math.h>
#include <stdlib.h>
#include <nlohmann/json.hpp>
#include <fstream>

namespace op
{
    template<typename TDatums>
    class WJsonOutput : public WorkerConsumer<TDatums>
    {
    public:
        explicit WJsonOutput(std::shared_ptr<op::VideoReader> videoReader, const std::string& jsonPath)
            : mVideoReader{ videoReader }
            , mJsonPath{ jsonPath }
        {
            mVideoWidth = mVideoReader->get(CV_CAP_PROP_FRAME_WIDTH);
            mVideoHeight = mVideoReader->get(CV_CAP_PROP_FRAME_HEIGHT);
            mVideoFPS = mVideoReader->get(CV_CAP_PROP_FPS);
            mVideoNumFrames = mVideoReader->get(CV_CAP_PROP_FRAME_COUNT);
        }

        ~WJsonOutput()
        {
            postProcess();
        }

        void initializationOnThread()
        {
        }

        void workConsumer(const TDatums& tDatums)
        {
            try
            {
                if (checkNoNullNorEmpty(tDatums))
                {
                    // Debugging log
                    dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

                    // Profiling speed
                    const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);

                    for (auto i = 0u; i < tDatums->size(); i++)
                    {
                        const auto& tDatum = (*tDatums)[i];

                        mRawFrames.push_back(FrameInfo(
                            tDatum.frameNumber,
                            std::vector<Array<float>>{
                            tDatum.poseKeypoints,
                                tDatum.faceKeypoints,
                                tDatum.handKeypoints[0],
                                tDatum.handKeypoints[1]
                        }
                        ));
                    }
                    // Profiling speed
                    Profiler::timerEnd(profilerKey);
                    Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
                    // Debugging log
                    dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                }
            }
            catch (const std::exception& e)
            {
                this->stop();
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

    private:
        //void postProcess(float confidenceThreshold = 0.1f, float mseLerpTheshold = 4.f,
        //    double mseHeadTranslationThreshold = 0.35, double mseHeadRotationThreshold = 0.03,
        //    int indexerInterval = 200);
        void postProcess(float confidenceThreshold = 0.1f, float mseLerpTheshold = 32.f,
            double mseHeadTranslationThreshold = 1.0, double mseHeadRotationThreshold = 0.1,
            int indexerInterval = 30, int maxFramesPerSegment = 10000);

        std::shared_ptr<op::VideoReader> mVideoReader;
        std::string mJsonPath;

        struct FrameInfo
        {
            int frameNumber;
            std::vector<Array<float>> keypoints;

            FrameInfo(int frameNumber_, std::vector<Array<float>>& keypoints_)
                : frameNumber{ frameNumber_ }
                , keypoints{ keypoints_ }
            {}
        };

        struct KeyFrame
        {
            int frameNumber;
            float x;
            float y;

            KeyFrame(int frameNumber_, float x_, float y_)
                : frameNumber{ frameNumber_ }
                , x{ x_ }
                , y{ y_ }
            {}
        };

        struct TransformRawFrame
        {
            int frameNumber;
            cv::Mat_<double> translation;
            cv::Mat_<double> rotation;

            TransformRawFrame(int frameNumber_,
                const cv::Mat& translation_,
                const cv::Mat& rotation_)
                : frameNumber{ frameNumber_ }
                , translation{ translation_ }
                , rotation{ rotation_ }
            {}
        };

        struct KeyFrame3D
        {
            int frameNumber;
            cv::Point3d value;

            KeyFrame3D(int frameNumber_,
                const cv::Point3d& value_)
                : frameNumber{ frameNumber_ }
                , value{ value_ }
            {}
        };

        struct FrameRaw
        {
            float confidence;
            float x;
            float y;

            FrameRaw(float confidence_, float x_, float y_)
                : confidence{ confidence_ }
                , x{ x_ }
                , y{ y_ }
            {}
        };

        struct PersonInfo
        {
            int startFrameNumber;
            int endFrameNumber;
            int numKeyFrames;

            PersonInfo()
                : startFrameNumber{ 0 }
                , endFrameNumber{ 0 }
                , numKeyFrames{ 0 }
            {
            }

            // frame number to raw frames
            std::map<int, std::vector<std::vector<FrameRaw>>> frames;

            // pose type to key frames
            std::map<int, std::vector<std::vector<std::vector<KeyFrame>>>> keypointTracks;

            std::vector<TransformRawFrame> headOrientationRawFrames;

            std::vector<std::vector<KeyFrame3D>> headRotationTracks;
            std::vector<std::vector<KeyFrame3D>> headTranslationTracks;
        };

        std::vector<FrameInfo> mRawFrames;

        double mVideoWidth;
        double mVideoHeight;
        double mVideoFPS;
        double mVideoNumFrames;

        DELETE_COPY(WJsonOutput);

        template<typename T>
        float squareLengthPoint3(const T& p)
        {
            return p.x*p.x + p.y*p.y + p.z*p.z;
        }

        template<typename T>
        float lengthPoint3(const T& p)
        {
            return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
        }

        template<typename T>
        float squareLengthPoint2(const T& p)
        {
            return p.x*p.x + p.y*p.y;
        }

        template<typename T>
        float lengthPoint2(const T& p)
        {
            return sqrt(p.x*p.x + p.y*p.y);
        }

        cv::Point2f normalizePoint2(const cv::Point2f& p)
        {
            auto length = lengthPoint2(p);
            return cv::Point2f(p.x / length, p.y / length);
        }
    };
}


// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    inline float rectArea(const std::tuple<float, float, float, float>& rect)
    {
        return (std::get<2>(rect) - std::get<0>(rect)) * (std::get<3>(rect) - std::get<1>(rect));
    }

    inline std::tuple<float, float, float, float> overlappingRect(const std::tuple<float, float, float, float>& rect1,
        const std::tuple<float, float, float, float>& rect2)
    {
        float left = std::max(std::get<0>(rect1), std::get<0>(rect2));
        float top = std::max(std::get<1>(rect1), std::get<1>(rect2));
        float right = std::min(std::get<2>(rect1), std::get<2>(rect2));
        float bottom = std::min(std::get<3>(rect1), std::get<3>(rect2));
        return std::tuple<float, float, float, float> {left, top, right, bottom};
    }

    COMPILE_TEMPLATE_DATUM(WJsonOutput);
}

#endif