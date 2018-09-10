#include <openpose/utilities/fileSystem.hpp>
#include <openpose/producer/videoReader.hpp>
#include <openpose/pose/poseParametersRender.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <fstream>
#include <math.h>

namespace op
{
    VideoReader::VideoReader(const std::string & videoPath, const unsigned int imageDirectoryStereo,
                             const std::string& cameraParameterPath, const std::string& videoInfoJsonPath, bool poseTrackingInfo) :
        VideoCaptureReader{videoPath, ProducerType::Video},
        mImageDirectoryStereo{imageDirectoryStereo},
        mPathName{getFileNameNoExtension(videoPath)},
        mIndexFramePoseTracking{0}
    {
        try
        {
            // If stereo setting --> load camera parameters
            if (imageDirectoryStereo > 1)
            {
                // Read camera parameters from SN
                auto serialNumbers = getFilesOnDirectory(cameraParameterPath, ".xml");
                // Security check
                if (serialNumbers.size() != mImageDirectoryStereo && mImageDirectoryStereo > 1)
                    error("Found different number of camera parameter files than the number indicated by"
                          " `--3d_views` ("
                          + std::to_string(serialNumbers.size()) + " vs. "
                          + std::to_string(mImageDirectoryStereo) + "). Make them equal or add"
                          + " `--3d_views 1`",
                          __LINE__, __FUNCTION__, __FILE__);
                // Get serial numbers
                for (auto& serialNumber : serialNumbers)
                    serialNumber = getFileNameNoExtension(serialNumber);
                // Get camera paremeters
                mCameraParameterReader.readParameters(cameraParameterPath, serialNumbers);
                // Set video size
                set(CV_CAP_PROP_FRAME_WIDTH, get(CV_CAP_PROP_FRAME_WIDTH)/mImageDirectoryStereo);
            }

            if (!videoInfoJsonPath.empty())
            {
                std::ifstream i(videoInfoJsonPath);
                json j;
                i >> j;
                mVideoInfo = std::make_shared<VideoInfo>(j);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void VideoReader::checkFrame()
    {
        try
        {
            if (mVideoInfo && mIndexFramePoseTracking < mVideoInfo->framesPoseTracking.size())
            {
                auto currentFrame = get(CV_CAP_PROP_POS_FRAMES);

                auto validFrame = mVideoInfo->framesPoseTracking[mIndexFramePoseTracking];

                if (validFrame != currentFrame)
                {
                    set(CV_CAP_PROP_POS_FRAMES, validFrame);
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::vector<cv::Mat> VideoReader::getCameraMatrices()
    {
        try
        {
            return mCameraParameterReader.getCameraMatrices();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<cv::Mat> VideoReader::getCameraExtrinsics()
    {
        try
        {
            return mCameraParameterReader.getCameraExtrinsics();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<cv::Mat> VideoReader::getCameraIntrinsics()
    {
        try
        {
            return mCameraParameterReader.getCameraIntrinsics();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::string VideoReader::getNextFrameName()
    {
        try
        {
            checkFrame();
            return mPathName + "_" + VideoCaptureReader::getNextFrameName();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    double VideoReader::get(const int capProperty)
    {
        try
        {
            if (capProperty == CV_CAP_PROP_FRAME_WIDTH)
                return VideoCaptureReader::get(capProperty) / mImageDirectoryStereo;
            else
                return VideoCaptureReader::get(capProperty);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.;
        }
    }

    void VideoReader::set(const int capProperty, const double value)
    {
        try
        {
            if (capProperty == CV_CAP_PROP_FRAME_WIDTH)
                return VideoCaptureReader::set(capProperty, value * mImageDirectoryStereo);
            else
                VideoCaptureReader::set(capProperty, value);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    cv::Mat VideoReader::getRawFrame()
    {
        try
        {
            checkFrame();
            ++mIndexFramePoseTracking;
            auto currentFrame = intRound(this->get(CV_CAP_PROP_POS_FRAMES));
            return VideoCaptureReader::getRawFrame();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return cv::Mat();
        }
    }

    std::vector<cv::Mat> VideoReader::getRawFrames()
    {
        try
        {
            auto cvMats = VideoCaptureReader::getRawFrames();
            // Split image
            if (cvMats.size() == 1 && mImageDirectoryStereo > 1)
            {
                cv::Mat cvMatConcatenated = cvMats.at(0);
                cvMats.clear();
                const auto individualWidth = cvMatConcatenated.cols/mImageDirectoryStereo;
                for (auto i = 0u ; i < mImageDirectoryStereo ; i++)
                    cvMats.emplace_back(
                        cv::Mat(cvMatConcatenated,
                                cv::Rect{(int)(i*individualWidth), 0,
                                         (int)individualWidth,
                                         (int)cvMatConcatenated.rows}));
            }
            // Security check
            else if (cvMats.size() != 1 && mImageDirectoryStereo > 1)
                error("Unexpected error. Notify us (" + std::to_string(mImageDirectoryStereo) + " vs. "
                      + std::to_string(mImageDirectoryStereo) + ").", __LINE__, __FUNCTION__, __FILE__);
            // Return images
            return cvMats;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }
}
