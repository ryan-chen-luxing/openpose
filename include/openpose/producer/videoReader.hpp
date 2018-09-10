#ifndef OPENPOSE_PRODUCER_VIDEO_READER_HPP
#define OPENPOSE_PRODUCER_VIDEO_READER_HPP

#include <openpose/3d/cameraParameterReader.hpp>
#include <openpose/core/common.hpp>
#include <openpose/producer/videoCaptureReader.hpp>
#include <nlohmann/json.hpp>

using namespace nlohmann;

namespace op
{
    struct VideoInfo
    {
        struct ClassificationInfo
        {
            struct FrameInfo
            {
                struct ObjectInfo
                {
                    //std::string classification;
                    float score;
                    double x1, x2, y1, y2;
                    int id;

                    explicit ObjectInfo(const json & j)
                    {
                        //classification = j["class"];
                        score = j["score"];
                        x1 = j["x1"];
                        x2 = j["x2"];
                        y1 = j["y1"];
                        y2 = j["y2"];
                        id = j["id"];
                    }
                };

                int frame;
                std::vector<std::shared_ptr<ObjectInfo>> objectsInfo;

                explicit FrameInfo(const json & j)
                {
                    frame = j["frame"];
                    for (auto objectInfo : j["persons"])
                    {
                        objectsInfo.push_back(std::shared_ptr<ObjectInfo>(new ObjectInfo(objectInfo)));
                    }
                }
            };

            std::vector<std::shared_ptr<FrameInfo>> framesInfo;

            explicit ClassificationInfo(const json & j)
            {
                for (auto frameInfo : j)
                {
                    framesInfo.push_back(std::shared_ptr<FrameInfo>(new FrameInfo(frameInfo)));
                }
            }
        };

        std::vector<int> framesPoseTracking;
        ClassificationInfo classificationInfo;

        explicit VideoInfo(const json & j)
            : classificationInfo{ j["classificationInfo"] }
        {
            framesPoseTracking = j["framesPoseTracking"];
        }
    };

    /**
     * VideoReader is a wrapper of the cv::VideoCapture class for video. It allows controlling a video (e.g. extracting
     * frames, setting resolution & fps, etc).
     */
    class OP_API VideoReader : public VideoCaptureReader
    {
    public:
        /**
         * Constructor of VideoReader. It opens the video as a wrapper of cv::VideoCapture. It includes a flag to
         * indicate whether the video should be repeated once it is completely read.
         * @param videoPath const std::string parameter with the full video path location.
         * @param imageDirectoryStereo const int parameter with the number of images per iteration (>1 would represent
         * stereo processing).
         * @param cameraParameterPath const std::string parameter with the folder path containing the camera
         * parameters (only required if imageDirectorystereo > 1).
         */
        explicit VideoReader(const std::string& videoPath, const unsigned int imageDirectoryStereo = 1,
                             const std::string& cameraParameterPath = "",
                             const std::string& videoInfoPath = "", bool poseTrackingInfo = false);

        std::vector<cv::Mat> getCameraMatrices();

        std::vector<cv::Mat> getCameraExtrinsics();

        std::vector<cv::Mat> getCameraIntrinsics();

        std::string getNextFrameName();

        double get(const int capProperty);

        void set(const int capProperty, const double value);

        std::shared_ptr<VideoInfo> getVideoInfo()
        {
            return mVideoInfo;
        }

        virtual bool shouldRelease()
        {
            return mVideoInfo && mIndexFramePoseTracking >= mVideoInfo->framesPoseTracking.size();
        }

    private:
        const unsigned int mImageDirectoryStereo;
        const std::string mPathName;
        CameraParameterReader mCameraParameterReader;

        std::shared_ptr<VideoInfo> mVideoInfo;
        int mIndexFramePoseTracking;

        cv::Mat getRawFrame();

        std::vector<cv::Mat> getRawFrames();

        void checkFrame();

        DELETE_COPY(VideoReader);
    };
}

#endif // OPENPOSE_PRODUCER_VIDEO_READER_HPP
