#ifndef OPENPOSE_FILESTREAM_W_JSON_OUTPUT_HPP
#define OPENPOSE_FILESTREAM_W_JSON_OUTPUT_HPP

#include <openpose/core/common.hpp>
#include <openpose/thread/workerConsumer.hpp>
#include <openpose/producer/videoReader.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <opencv2/opencv.hpp>
#include <map>
#include <math.h>
#include <stdlib.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <algorithm>
#include <limits>
#include <sstream>

namespace op
{
    template<typename TDatums>
    class OP_API WJsonOutput : public WorkerConsumer<TDatums>
    {
    public:
        explicit WJsonOutput(PoseModel poseModel, std::shared_ptr<op::VideoReader> videoReader, const std::string& jsonPath)
            : mPoseModel{ poseModel }
            , mVideoReader{ videoReader }
            , mJsonPath{ jsonPath }
        {
            mVideoWidth = std::size_t(mVideoReader->get(CV_CAP_PROP_FRAME_WIDTH));
            mVideoHeight = std::size_t(mVideoReader->get(CV_CAP_PROP_FRAME_HEIGHT));
            mVideoFPS = std::size_t(mVideoReader->get(CV_CAP_PROP_FPS));
            mVideoNumFrames = std::size_t(mVideoReader->get(CV_CAP_PROP_FRAME_COUNT));
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

                    for (std::size_t i = 0; i < tDatums->size(); i++)
                    {
                        const auto& tDatum = (*tDatums)[i];

                        std::vector<Array<float>> keypoints{
                            tDatum.poseKeypoints,
                            tDatum.faceKeypoints,
                            tDatum.handKeypoints[0],
                            tDatum.handKeypoints[1]
                        };

                        mRawFrames.push_back(FrameInfo(
                            tDatum.frameNumber, keypoints));
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
        struct FrameInfo
        {
            std::size_t frameNumber;
            std::vector<Array<float>> keypoints;

            FrameInfo(std::size_t frameNumber_, const std::vector<Array<float>>& keypoints_)
                : frameNumber{ frameNumber_ }
                , keypoints{ keypoints_ }
            {}
        };

        struct KeyFrame
        {
            std::size_t frameNumber;
            float x;
            float y;

            KeyFrame(std::size_t frameNumber_, float x_, float y_)
                : frameNumber{ frameNumber_ }
                , x{ x_ }
                , y{ y_ }
            {}
        };

        struct TransformRawFrame
        {
            std::size_t frameNumber;
            cv::Mat_<double> translation;
            cv::Mat_<double> rotation;

            TransformRawFrame(std::size_t frameNumber_,
                const cv::Mat& translation_,
                const cv::Mat& rotation_)
                : frameNumber{ frameNumber_ }
                , translation{ translation_ }
                , rotation{ rotation_ }
            {}
        };

        struct KeyFrame3D
        {
            std::size_t frameNumber;
            cv::Point3d value;

            KeyFrame3D(std::size_t frameNumber_,
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
            std::size_t startFrameNumber;
            std::size_t endFrameNumber;
            std::size_t numKeyFrames;

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

        std::size_t mVideoWidth;
        std::size_t mVideoHeight;
        std::size_t mVideoFPS;
        std::size_t mVideoNumFrames;

        std::shared_ptr<op::VideoReader> mVideoReader;
        std::string mJsonPath;
        PoseModel mPoseModel;

        DELETE_COPY(WJsonOutput);

        template<typename T>
        float squareLengthPoint3f(const T& p)
        {
            return p.x*p.x + p.y*p.y + p.z*p.z;
        }

        template<typename T>
        float lengthPoint3f(const T& p)
        {
            return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
        }

        template<typename T>
        float squareLengthPoint2f(const T& p)
        {
            return p.x*p.x + p.y*p.y;
        }

        template<typename T>
        float lengthPoint2f(const T& p)
        {
            return sqrt(p.x*p.x + p.y*p.y);
        }

        template<typename T>
        inline double squareLengthPoint3d(const T& p)
        {
            return p.x*p.x + p.y*p.y + p.z*p.z;
        }

        template<typename T>
        double lengthPoint3d(const T& p)
        {
            return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
        }

        template<typename T>
        inline double squareLengthPoint2d(const T& p)
        {
            return p.x*p.x + p.y*p.y;
        }

        template<typename T>
        double lengthPoint2d(const T& p)
        {
            return sqrt(p.x*p.x + p.y*p.y);
        }

        cv::Point2f normalizePoint2f(const cv::Point2f& p)
        {
            auto length = lengthPoint2f(p);
            return cv::Point2f(p.x / length, p.y / length);
        }

        float rectArea(const std::tuple<float, float, float, float>& rect)
        {
            return (std::get<2>(rect) - std::get<0>(rect)) * (std::get<3>(rect) - std::get<1>(rect));
        }

        std::tuple<float, float, float, float> overlappingRect(
            const std::tuple<float, float, float, float>& rect1,
            const std::tuple<float, float, float, float>& rect2)
        {
            float left = std::max(std::get<0>(rect1), std::get<0>(rect2));
            float top = std::max(std::get<1>(rect1), std::get<1>(rect2));
            float right = std::min(std::get<2>(rect1), std::get<2>(rect2));
            float bottom = std::min(std::get<3>(rect1), std::get<3>(rect2));
            return std::tuple<float, float, float, float> {left, top, right, bottom};
        }

        template<typename T>
        std::string toString(const T& value)
        {
            std::stringstream ss;
            ss << value;
            return ss.str();
        }

        //void postProcess(float confidenceThreshold = 0.1f, float mseLerpTheshold = 4.f,
        //    double mseHeadTranslationThreshold = 0.35, double mseHeadRotationThreshold = 0.03,
        //    int indexerInterval = 200);
        void postProcess(float confidenceThreshold = 0.1f,
            float mseLerpTheshold = 48.f,
            double mseHeadTranslationThreshold = 0.6,
            double mseHeadRotationThreshold = 0.06,
            std::size_t maxFramesPerSegment = 0,
            bool personByPerson = true,
            bool frameByFrame = true,
            bool outputMetafile = true)
        {
            // processing raw frames into skeletons
            std::vector<PersonInfo> personsRaw;
            std::vector<std::size_t> personsBeingTracked;
            for (const auto& rawframe : this->mRawFrames)
            {
                std::size_t numberPeople = 0;
                for (std::size_t vectorIndex = 0u; vectorIndex < rawframe.keypoints.size(); vectorIndex++)
                {
                    numberPeople = fastMax(numberPeople, std::size_t(rawframe.keypoints[vectorIndex].getSize(0)));
                }

                if (!personsBeingTracked.empty())
                {
                    const auto& personTracked = personsRaw[personsBeingTracked.front()];
                    if (personTracked.endFrameNumber != rawframe.frameNumber - 1)
                    {
                        // detect frame jump
                        personsBeingTracked.clear();
                    }
                }

                std::vector<std::size_t> personsMatched;
                std::vector<std::size_t> newPersonsIdentified;
                for (std::size_t personIndex = 0; personIndex < numberPeople; personIndex++)
                {
                    const auto personRectangle = getKeypointsRectangle(rawframe.keypoints[0], personIndex, confidenceThreshold);
                    if (personRectangle.area() > 0)
                    {
                        std::vector<std::vector<FrameRaw>> rawFrames;
                        for (std::size_t vectorIndex = 0; vectorIndex < rawframe.keypoints.size(); vectorIndex++)
                        {
                            const auto& keypoints = rawframe.keypoints[vectorIndex];

                            const auto numberElementsPerRaw = keypoints.getSize(1) * keypoints.getSize(2);

                            std::vector<FrameRaw> frames;

                            if (numberElementsPerRaw > 0)
                            {
                                const auto finalIndex = personIndex * numberElementsPerRaw;

                                for (std::size_t part = 0; part < keypoints.getSize(1); part++)
                                {
                                    auto confidence = keypoints[finalIndex + part * keypoints.getSize(2) + 2];
                                    auto x = keypoints[finalIndex + part * keypoints.getSize(2)];
                                    auto y = keypoints[finalIndex + part * keypoints.getSize(2) + 1];
                                    frames.push_back(FrameRaw(confidence, x, y));
                                }
                            }

                            rawFrames.push_back(frames);
                        }

                        auto skeletonMatched = false;
                        if (!personsBeingTracked.empty())
                        {
                            // try to match a skeleton being tracked
                            auto minDifference = std::numeric_limits<float>::max();
                            std::size_t personBestMatched = std::numeric_limits<std::size_t>::max();
                            for (auto i : personsBeingTracked)
                            {
                                // check if the person has been matched already in the personsMatched array.
                                if (std::find(personsMatched.begin(), personsMatched.end(), i) != personsMatched.end())
                                    continue;

                                float sumDifference = 0.f;

                                const auto& personTracked = personsRaw[i];

                                const auto& rawFramesTracked = personTracked.frames.at(rawframe.frameNumber - 1);

                                auto numValidParts = 0;
                                for (std::size_t trackingType = 0; trackingType < rawFrames.size(); ++trackingType)
                                {
                                    for (std::size_t part = 0; part < rawFrames[trackingType].size(); ++part)
                                    {
                                        const auto rawFrame = rawFrames[trackingType][part];
                                        const auto rawFrameTracked = rawFramesTracked[trackingType][part];

                                        if (rawFrame.confidence >= confidenceThreshold &&
                                            rawFrameTracked.confidence >= confidenceThreshold)
                                        {
                                            auto difference = abs(rawFrame.x - rawFrameTracked.x) + abs(rawFrame.y - rawFrameTracked.y);

                                            sumDifference += difference;

                                            numValidParts++;
                                        }
                                    }
                                }

                                if (numValidParts > 0)
                                {
                                    sumDifference /= numValidParts;

                                    if (sumDifference < minDifference)
                                    {
                                        minDifference = sumDifference;
                                        personBestMatched = i;
                                    }
                                }
                            }

                            if (minDifference < (personRectangle.width + personRectangle.height))
                            {
                                // match a skeleton
                                auto& person = personsRaw[personBestMatched];
                                person.endFrameNumber = rawframe.frameNumber;
                                person.frames[rawframe.frameNumber] = rawFrames;
                                personsMatched.push_back(personBestMatched);
                                skeletonMatched = true;
                            }
                        }

                        if (!skeletonMatched)
                        {
                            PersonInfo person;
                            person.startFrameNumber = rawframe.frameNumber;
                            person.endFrameNumber = rawframe.frameNumber;
                            person.frames[rawframe.frameNumber] = rawFrames;
                            personsRaw.push_back(person);
                            newPersonsIdentified.push_back(personsRaw.size() - 1);
                        }
                    }
                }

                personsBeingTracked.erase(std::remove_if(personsBeingTracked.begin(), personsBeingTracked.end(), [&](std::size_t index)
                {
                    return std::find(personsMatched.begin(), personsMatched.end(), index) == personsMatched.end();
                }), personsBeingTracked.end());

                personsBeingTracked.insert(personsBeingTracked.end(), newPersonsIdentified.begin(), newPersonsIdentified.end());
            }

            // matching the skeleton with classification infos
            std::vector<PersonInfo*> persons;
            for (auto& personRaw : personsRaw)
            {
                // filter out noise
                if (personRaw.frames.size() > 1)
                {
                    if (mVideoReader->getVideoInfo())
                    {
                        auto classificationItr = mVideoReader->getVideoInfo()->classificationInfo.framesInfo.begin();
                        unsigned int numFramesMatched = 0;
                        for (auto kvprf : personRaw.frames)
                        {
                            while ((*classificationItr)->frame < kvprf.first)
                            {
                                classificationItr++;
                            }

                            Rectangle<float> personRawRectangle{};
                            {
                                float minX = std::numeric_limits<float>::max();
                                float maxX = 0.f;
                                float minY = minX;
                                float maxY = maxX;
                                for (const auto& part : kvprf.second.front())
                                {
                                    if (part.confidence > confidenceThreshold)
                                    {
                                        // Set X
                                        if (maxX < part.x)
                                            maxX = part.x;
                                        if (minX > part.x)
                                            minX = part.x;
                                        // Set Y
                                        if (maxY < part.y)
                                            maxY = part.y;
                                        if (minY > part.y)
                                            minY = part.y;
                                    }
                                }

                                if (maxX >= minX && maxY >= minY)
                                    personRawRectangle = Rectangle<float>{ minX, minY, maxX - minX, maxY - minY };
                            }

                            std::tuple<float, float, float, float> personRect
                            {
                                personRawRectangle.x, personRawRectangle.y,
                                personRawRectangle.x + personRawRectangle.width,
                                personRawRectangle.y + personRawRectangle.height
                            };

                            // search for best matched person
                            float areaOverlappingPercentageBest = 0.6f;
                            auto personBestMatched = (*classificationItr)->objectsInfo.end();

                            for (auto objectInfoItr = (*classificationItr)->objectsInfo.begin();
                                objectInfoItr != (*classificationItr)->objectsInfo.end(); ++objectInfoItr)
                            {
                                const float x1{ float((*objectInfoItr)->x1 * this->mVideoWidth) };
                                const float y1{ float((*objectInfoItr)->y1 * this->mVideoHeight) };
                                const float x2{ float((*objectInfoItr)->x2 * this->mVideoWidth) };
                                const float y2{ float((*objectInfoItr)->y2 * this->mVideoHeight) };

                                const std::tuple<float, float, float, float> identityRect{ x1, y1, x2, y2 };
                                const auto areaIdentity = rectArea(identityRect);


                                auto rectOverlapping = overlappingRect(personRect, identityRect);

                                auto areaPerson = rectArea(personRect);
                                auto areaOverlapping = rectArea(rectOverlapping);

                                float areaOverlappingPercentageAvg = (areaOverlapping / areaPerson + areaOverlapping / areaIdentity) * 0.5f;

                                if (areaOverlappingPercentageAvg > areaOverlappingPercentageBest)
                                {
                                    areaOverlappingPercentageBest = areaOverlappingPercentageAvg;
                                    personBestMatched = objectInfoItr;
                                }
                            }

                            if (personBestMatched != (*classificationItr)->objectsInfo.end())
                            {
                                numFramesMatched++;
                            }
                        }

                        if (float(numFramesMatched) / (personRaw.endFrameNumber - personRaw.startFrameNumber + 1) > 0.5)
                        {
                            persons.push_back(&personRaw);
                        }
                    }
                    else
                    {
                        // push all the persons without classification info
                        persons.push_back(&personRaw);
                    }
                }
            }

            // generate face vectors
            const std::vector<std::size_t> keypointsPoseEstimationIndicesBody25
            {
                0, 15, 16, 17, 18
            };

            const std::vector<std::size_t> keypointsPoseEstimationIndicesCoco
            {
                0, 14, 15, 16, 17
            };

            std::vector<std::size_t> keypointsPoseEstimationIndices;
            if (mPoseModel == PoseModel::BODY_25)
            {
                keypointsPoseEstimationIndices = keypointsPoseEstimationIndicesBody25;
            }
            else if (mPoseModel == PoseModel::COCO_18)
            {
                keypointsPoseEstimationIndices = keypointsPoseEstimationIndicesCoco;
            }

            // 3D model points.
            const std::vector<cv::Point3d> keypointsReferencePose
            {
                cv::Point3d(0, 0, 0),
                cv::Point3d(-0.224, 0.209, -0.261),
                cv::Point3d(0.224, 0.209, -0.261),
                cv::Point3d(-0.644, 0.130, -1),
                cv::Point3d(0.644, 0.130, -1)
            };

            const std::vector<std::size_t> keypointsFaceEstimationIndices
            {
                //36, 45, 39, 42, 
                8, 19, 24
            };

            // 3D model points.
            const std::vector<cv::Point3d> keypointsReferenceFace
            {
                //cv::Point3d(-25.0f, 17.0f, -13.7f),
                //cv::Point3d(25.0f, 17.0f, -13.7f),
                //cv::Point3d(-15.0f, 17.0f, -13.5f),
                //cv::Point3d(15.0f, 17.0f, -13.5f),
                cv::Point3d(0, -0.677, -0.260),
                cv::Point3d(-0.251, 0.310, -0.230),
                cv::Point3d(0.251, 0.310, -0.230),
            };

            for (auto& p : persons)
            {
                auto& person = *p;

                if (person.frames.empty())
                    continue;

                TransformRawFrame* headOrientationFrameLast = nullptr;
                for (const auto& kvrf : person.frames)
                {
                    auto frameNumber = kvrf.first;
                    if (frameNumber < person.startFrameNumber ||
                        frameNumber > person.endFrameNumber)
                    {
                        continue;
                    }

                    //if (headOrientationFrameLast != nullptr &&
                    //    frameNumber != headOrientationFrameLast->frameNumber + 1)
                    //{
                    //    op::error("headOrientationFrameLast != nullptr && frameNumber != headOrientationFrameLast->frameNumber + 1", __LINE__, __FUNCTION__, __FILE__);
                    //}

                    const auto& rawFrames = kvrf.second[0];

                    // 2D image points.
                    std::vector<cv::Point2d> image_points;
                    std::vector<cv::Point3d> model_points;
                    for (std::size_t i = 0; i < keypointsPoseEstimationIndices.size(); i++)
                    {
                        auto part = keypointsPoseEstimationIndices[i];

                        const auto& rawFrame = rawFrames[part];

                        if (rawFrame.confidence > confidenceThreshold)
                        {
                            image_points.push_back(cv::Point2d(rawFrame.x, rawFrame.y));
                            model_points.push_back(keypointsReferencePose[i]);
                        }
                    }

                    if (!kvrf.second[1].empty())
                    {
                        const auto& rawFramesFace = kvrf.second[1];
                        for (std::size_t i = 0; i < keypointsFaceEstimationIndices.size(); i++)
                        {
                            auto part = keypointsFaceEstimationIndices[i];

                            const auto& rawFrame = rawFramesFace[part];

                            if (rawFrame.confidence > confidenceThreshold)
                            {
                                image_points.push_back(cv::Point2d(rawFrame.x, rawFrame.y));
                                model_points.push_back(keypointsReferenceFace[i]);
                            }
                        }
                    }

                    if (image_points.size() >= 4)
                    {
                        // Camera internals
                        double focal_length = mVideoWidth; // Approximate focal length.
                        cv::Point2d center = cv::Point2d(mVideoWidth / 2, mVideoHeight / 2);
                        cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
                        // Assuming no lens distortion
                        cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);

                        // Output rotation and translation
                        cv::Mat translation_vector;
                        cv::Mat rotation_vector;

                        if (headOrientationFrameLast != nullptr)
                        {
                            translation_vector = headOrientationFrameLast->translation;
                            rotation_vector = headOrientationFrameLast->rotation;
                        }

                        // Solve for pose
                        cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                            rotation_vector, translation_vector, headOrientationFrameLast != nullptr, cv::SOLVEPNP_ITERATIVE);

                        cv::Mat translationResult;
                        cv::Mat rotationResult;

                        rotation_vector.copyTo(rotationResult);
                        translation_vector.copyTo(translationResult);
                        TransformRawFrame trf(frameNumber, translationResult, rotationResult);
                        person.headOrientationRawFrames.push_back(trf);

                        headOrientationFrameLast = &person.headOrientationRawFrames.back();
                    }
                }
            }

            // processing raw frames into key frames
            for (auto& p : persons)
            {
                auto& person = *p;

                const std::size_t nRawFrameTypes = person.frames.begin()->second.size();

                for (std::size_t iRawFrameType = 0; iRawFrameType < nRawFrameTypes; ++iRawFrameType)
                {
                    std::vector<std::vector<std::vector<KeyFrame>>> keypointTracks;
                    const std::size_t nKeypoints = person.frames.begin()->second[iRawFrameType].size();

                    bool validTrack = false;
                    for (std::size_t iKeypointId = 0; iKeypointId < nKeypoints; ++iKeypointId)
                    {
                        std::vector<std::vector<KeyFrame>> keyFrameTracks;

                        int lastKeyframeNumber = -1;
                        int lastFrameNumber = -1;

                        keyFrameTracks.push_back(std::vector<KeyFrame>{});
                        for (const auto& kvrf : person.frames)
                        {
                            auto frameNumber = kvrf.first;
                            if (frameNumber < person.startFrameNumber ||
                                frameNumber > person.endFrameNumber)
                            {
                                continue;
                            }

                            const auto& rawFrame = kvrf.second[iRawFrameType][iKeypointId];

                            if (rawFrame.confidence > confidenceThreshold)
                            {
                                validTrack = true;
                                if (lastKeyframeNumber < 0 ||
                                    frameNumber == person.endFrameNumber)
                                {
                                    // push the first and the last frame
                                    keyFrameTracks.back().push_back(KeyFrame(frameNumber, rawFrame.x, rawFrame.y));
                                    lastKeyframeNumber = frameNumber;
                                }
                                else if (lastFrameNumber != frameNumber - 1 ||
                                    // this will make sure keyframes will not span across segments
                                    (!personByPerson && maxFramesPerSegment > 0 && (lastFrameNumber + 1) % maxFramesPerSegment == 0))
                                {
                                    // detect discontinuity on frames
                                    // push last frame
                                    if (lastKeyframeNumber != lastFrameNumber)
                                    {
                                        const auto& lastFrame = person.frames[lastFrameNumber][iRawFrameType][iKeypointId];
                                        keyFrameTracks.back().push_back(KeyFrame(lastFrameNumber, lastFrame.x, lastFrame.y));
                                    }

                                    // push this frame
                                    keyFrameTracks.push_back(std::vector<KeyFrame>{});
                                    keyFrameTracks.back().push_back(KeyFrame(frameNumber, rawFrame.x, rawFrame.y));

                                    lastKeyframeNumber = frameNumber;
                                }
                                else
                                {
                                    const auto& lastKeyFrame = person.frames[lastKeyframeNumber][iRawFrameType][iKeypointId];
                                    cv::Point2f lastKeyFramePos(lastKeyFrame.x, lastKeyFrame.y);

                                    cv::Point2f currentFramePos = cv::Point2f(rawFrame.x, rawFrame.y);

                                    float mse = 0.f;
                                    int nFrames = 0;
                                    int lastValidFrameNumber = 0;
                                    for (auto f = frameNumber - 1; f > lastKeyframeNumber; --f)
                                    {
                                        if (person.frames.find(f) != person.frames.end())
                                        {
                                            const auto& lastFrame = person.frames[f][iRawFrameType][iKeypointId];
                                            if (lastFrame.confidence > 0.f)
                                            {
                                                if (nFrames == 0)
                                                {
                                                    lastValidFrameNumber = f;
                                                }

                                                cv::Point2f lastFramePos(lastFrame.x, lastFrame.y);

                                                float t = float(f - lastKeyframeNumber) / float(frameNumber - lastKeyframeNumber);
                                                cv::Point2f estimatedPos = lastKeyFramePos * (1 - t) + currentFramePos * t;

                                                mse += squareLengthPoint2f(estimatedPos - lastFramePos);
                                                nFrames++;
                                            }
                                        }
                                    }

                                    if (nFrames > 0)
                                    {
                                        mse /= nFrames;

                                        if (mse > mseLerpTheshold)
                                        {
                                            // push last frame
                                            const auto& lastFrame = person.frames[lastValidFrameNumber][iRawFrameType][iKeypointId];
                                            keyFrameTracks.back().push_back(KeyFrame(lastValidFrameNumber, lastFrame.x, lastFrame.y));
                                            lastKeyframeNumber = lastValidFrameNumber;
                                        }
                                    }
                                }

                                lastFrameNumber = frameNumber;
                            }
                        }

                        for (const auto& kft : keyFrameTracks)
                        {
                            person.numKeyFrames += kft.size();
                        }

                        keypointTracks.push_back(keyFrameTracks);
                    }

                    if (validTrack)
                    {
                        person.keypointTracks[iRawFrameType] = keypointTracks;
                    }
                }
            }

            for (auto& p : persons)
            {
                auto& person = *p;

                if (person.frames.empty())
                    continue;

                person.headTranslationTracks.push_back(std::vector<KeyFrame3D>{});
                person.headRotationTracks.push_back(std::vector<KeyFrame3D>{});
                auto lastKeyframe = person.headOrientationRawFrames.end();
                auto lastFrame = person.headOrientationRawFrames.end();
                for (auto itrf = person.headOrientationRawFrames.begin();
                    itrf != person.headOrientationRawFrames.end(); ++itrf)
                {
                    auto frameNumber = itrf->frameNumber;
                    cv::Point3d translation{
                        itrf->translation(0),
                        itrf->translation(1),
                        itrf->translation(2)
                    };
                    cv::Point3d rotation{
                        itrf->rotation(0),
                        itrf->rotation(1),
                        itrf->rotation(2)
                    };

                    if (itrf == person.headOrientationRawFrames.begin() ||
                        itrf == person.headOrientationRawFrames.end() - 1)
                    {
                        // push the first and the last frame
                        person.headTranslationTracks.back().push_back(KeyFrame3D(frameNumber, translation));
                        person.headRotationTracks.back().push_back(KeyFrame3D(frameNumber, rotation));
                        lastKeyframe = itrf;
                    }
                    else if (lastFrame->frameNumber != frameNumber - 1 ||
                        // this will make sure keyframes will not span across segments
                        (!personByPerson && maxFramesPerSegment > 0 && (lastFrame->frameNumber + 1) % maxFramesPerSegment == 0))
                    {
                        // detect discontinuity on frames
                        // push last frame
                        if (lastKeyframe->frameNumber != lastFrame->frameNumber)
                        {
                            cv::Point3d lastTranslation
                            {
                                lastFrame->translation(0),
                                lastFrame->translation(1),
                                lastFrame->translation(2)
                            };
                            cv::Point3d lastRotation
                            {
                                lastFrame->rotation(0),
                                lastFrame->rotation(1),
                                lastFrame->rotation(2)
                            };

                            person.headTranslationTracks.back().push_back(KeyFrame3D(lastFrame->frameNumber, lastTranslation));
                            person.headRotationTracks.back().push_back(KeyFrame3D(lastFrame->frameNumber, lastRotation));
                        }

                        person.headTranslationTracks.push_back(std::vector<KeyFrame3D>{});
                        person.headRotationTracks.push_back(std::vector<KeyFrame3D>{});
                        // push this frame
                        person.headTranslationTracks.back().push_back(KeyFrame3D(frameNumber, translation));
                        person.headRotationTracks.back().push_back(KeyFrame3D(frameNumber, rotation));

                        lastKeyframe = itrf;
                    }
                    else
                    {
                        cv::Point3d lastKeyTranslation
                        {
                            lastKeyframe->translation(0),
                            lastKeyframe->translation(1),
                            lastKeyframe->translation(2)
                        };
                        cv::Point3d lastKeyRotation
                        {
                            lastKeyframe->rotation(0),
                            lastKeyframe->rotation(1),
                            lastKeyframe->rotation(2)
                        };


                        double mseTranslation = 0.0;
                        double mseRotation = 0.0;
                        std::size_t nFrames = 0;
                        for (auto i = lastKeyframe + 1; i != itrf; ++i)
                        {
                            cv::Point3d tt
                            {
                                i->translation(0),
                                i->translation(1),
                                i->translation(2)
                            };
                            cv::Point3d rr
                            {
                                i->rotation(0),
                                i->rotation(1),
                                i->rotation(2)
                            };

                            double t = double(i->frameNumber - lastKeyframe->frameNumber) / double(frameNumber - lastKeyframe->frameNumber);
                            cv::Point3d estimatedTranslation = lastKeyTranslation * (1 - t) + translation * t;
                            cv::Point3d estimatedRotation = lastKeyRotation * (1 - t) + rotation * t;

                            mseTranslation += squareLengthPoint3d(estimatedTranslation - tt);
                            mseRotation += squareLengthPoint3d(estimatedRotation - rr);
                            nFrames++;
                        }

                        if (nFrames > 0)
                        {
                            mseTranslation /= nFrames;
                            mseRotation /= nFrames;

                            if (mseTranslation > mseHeadTranslationThreshold)
                            {
                                cv::Point3d lastTranslation
                                {
                                    lastFrame->translation(0),
                                    lastFrame->translation(1),
                                    lastFrame->translation(2)
                                };

                                person.headTranslationTracks.back().push_back(KeyFrame3D(lastFrame->frameNumber, lastTranslation));
                            }

                            if (mseRotation > mseHeadRotationThreshold)
                            {
                                cv::Point3d lastRotation
                                {
                                    lastFrame->rotation(0),
                                    lastFrame->rotation(1),
                                    lastFrame->rotation(2)
                                };

                                person.headRotationTracks.back().push_back(KeyFrame3D(lastFrame->frameNumber, lastRotation));
                            }
                        }
                    }

                    lastFrame = itrf;
                }

                for (const auto& track : person.headTranslationTracks)
                {
                    person.numKeyFrames += track.size();
                }

                for (const auto& track : person.headRotationTracks)
                {
                    person.numKeyFrames += track.size();
                }
            }

            // output JSON string
            if (personByPerson)
            {
                std::vector<nlohmann::json> jRoots;
                std::size_t iPerson = 0;
                while (iPerson < persons.size())
                {
                    jRoots.push_back(nlohmann::json{});
                    nlohmann::json jPersons;
                    std::size_t numKeyFrames = 0;
                    std::vector<PersonInfo*> personsThisSegment;
                    std::size_t startFrameNumberSegment = std::numeric_limits<std::size_t>::max();
                    std::size_t endFrameNumberSegment = 0;
                    for (; iPerson < persons.size(); ++iPerson)
                    {
                        auto& pp = persons[iPerson];
                        auto& p = *pp;

                        nlohmann::json jPerson;
                        jPerson["id"] = iPerson;
                        jPerson["startFrame"] = p.startFrameNumber;
                        jPerson["endFrame"] = p.endFrameNumber;

                        startFrameNumberSegment = std::min(startFrameNumberSegment, p.startFrameNumber);
                        endFrameNumberSegment = std::max(endFrameNumberSegment, p.endFrameNumber);
                        for (const auto& kvt : p.keypointTracks)
                        {
                            if (!kvt.second.empty())
                            {
                                //nlohmann::json jTracks;

                                std::string keypointName;
                                switch (kvt.first)
                                {
                                case 0:
                                    keypointName = "pose";
                                    break;
                                case 1:
                                    keypointName = "face";
                                    break;
                                case 2:
                                    keypointName = "handL";
                                    break;
                                case 3:
                                    keypointName = "handR";
                                    break;
                                default:
                                    break;
                                }

                                //jTracks["type"] = keypointName;

                                nlohmann::json jKeypoints;
                                for (std::size_t iKeypoint = 0; iKeypoint < kvt.second.size(); ++iKeypoint)
                                {
                                    nlohmann::json jKeyFrameTracks;
                                    for (const auto& kft : kvt.second[iKeypoint])
                                    {
                                        if (kft.size() > 1)
                                        {
                                            nlohmann::json jKeyFrames;
                                            for (const auto& kf : kft)
                                            {
                                                nlohmann::json keyFrame;
                                                keyFrame["n"] = kf.frameNumber;
                                                keyFrame["x"] = int(round(kf.x));
                                                keyFrame["y"] = int(round(kf.y));
                                                jKeyFrames.push_back(keyFrame);
                                            }
                                            jKeyFrameTracks.push_back(jKeyFrames);
                                        }
                                    }

                                    if (!jKeyFrameTracks.empty())
                                    {
                                        jKeypoints[toString(iKeypoint)] = jKeyFrameTracks;
                                    }
                                }

                                //jTracks["keypoints"] = jKeypoints;
                                //jPerson["tracks"].push_back(jTracks);
                                jPerson["tracks"][keypointName] = jKeypoints;
                            }
                        }

                        nlohmann::json jFaceTranslationTracks;
                        for (const auto& htt : p.headTranslationTracks)
                        {
                            if (!htt.empty())
                            {
                                nlohmann::json jTrack;
                                for (const auto& htkf : htt)
                                {
                                    nlohmann::json jKeyframe3D;
                                    jKeyframe3D["n"] = htkf.frameNumber;
                                    std::vector<int> value{
                                        int(round(htkf.value.x * 1000)),
                                        int(round(htkf.value.y * 1000)),
                                        int(round(htkf.value.z * 1000))
                                    };
                                    jKeyframe3D["v"] = value;
                                    jTrack.push_back(jKeyframe3D);
                                }

                                jFaceTranslationTracks.push_back(jTrack);
                            }
                        }
                        //jPerson["faceTranslation"] = jFaceTranslationTracks;
                        if (!jFaceTranslationTracks.empty())
                        {
                            jPerson["tracks"]["ht"] = jFaceTranslationTracks;
                        }

                        nlohmann::json jFaceRotationTracks;
                        for (const auto& hrt : p.headRotationTracks)
                        {
                            if (!hrt.empty())
                            {
                                nlohmann::json jTrack;
                                for (const auto& hrkf : hrt)
                                {
                                    nlohmann::json jKeyframe3D;
                                    jKeyframe3D["n"] = hrkf.frameNumber;
                                    std::vector<int> value{
                                        int(round(hrkf.value.x * 1000)),
                                        int(round(hrkf.value.y * 1000)),
                                        int(round(hrkf.value.z * 1000))
                                    };
                                    jKeyframe3D["v"] = value;
                                    jTrack.push_back(jKeyframe3D);
                                }

                                jFaceRotationTracks.push_back(jTrack);
                            }
                        }
                        //jPerson["faceRotation"] = jFaceRotationTracks;
                        if (!jFaceRotationTracks.empty())
                        {
                            jPerson["tracks"]["hr"] = jFaceRotationTracks;
                        }

                        jPersons.push_back(jPerson);

                        personsThisSegment.push_back(pp);

                        numKeyFrames += p.numKeyFrames;

                        if ((maxFramesPerSegment != 0 && numKeyFrames >= maxFramesPerSegment) ||
                            iPerson >= persons.size() - 1)
                        {
                            jRoots.back()["persons"] = jPersons;

                            ++iPerson;
                            break;
                        }
                    }
                }

                for (std::size_t i = 0; i < jRoots.size(); ++i)
                {
                    auto& jRoot = jRoots[i];
                    jRoot["numSegments"] = jRoots.size();
                    jRoot["segmentIndex"] = i;
                    std::stringstream ss;
                    ss << this->mJsonPath << "_pbp_" << i;
                    std::ofstream ofile(ss.str() + ".json");
                    ofile << jRoot;
                    ofile.close();
#ifdef _WIN32
                    std::string command = "7z.exe a " + ss.str() + ".7z " + ss.str() + ".json";
                    system(command.c_str());
#endif
                }
            }
            
            if (frameByFrame)
            {
                std::vector<nlohmann::json> jRoots;
                struct KeyFrameFlat
                {
                    std::size_t frameNumber;
                    std::size_t personId;
                    std::size_t keypointId;
                    bool lastOne;
                    float x;
                    float y;

                    KeyFrameFlat(std::size_t frameNumber_,
                        std::size_t personId_,
                        std::size_t keypointId_,
                        bool lastOne_,
                        float x_, float y_)
                        : frameNumber{ frameNumber_ }
                        , personId{ personId_ }
                        , keypointId{ keypointId_ }
                        , lastOne{ lastOne_ }
                        , x{ x_ }
                        , y{ y_ }
                    {}
                };

                struct KeyFrame3DFlat
                {
                    std::size_t frameNumber;
                    std::size_t personId;
                    bool lastOne;
                    cv::Point3d value;

                    KeyFrame3DFlat(std::size_t frameNumber_,
                        std::size_t personId_,
                        bool lastOne_,
                        const cv::Point3d& value_)
                        : frameNumber{ frameNumber_ }
                        , personId{ personId_ }
                        , lastOne{ lastOne_ }
                        , value{ value_ }
                    {}
                };

                // frameNumber->keyframe type->person->keyframes
                std::map<std::size_t, std::map<std::size_t, std::map<std::size_t, std::vector<KeyFrameFlat>>>> keyframesFlat;
                std::map<std::size_t, std::map<std::size_t, std::map<std::size_t, std::vector<KeyFrame3DFlat>>>> keyframes3DFlat;
                
                std::size_t iPerson = 0;
                for (; iPerson < persons.size(); ++iPerson)
                {
                    auto& pp = persons[iPerson];
                    auto& person = *pp;

                    for (auto kvt : person.keypointTracks)
                    {
                        auto poseType = kvt.first;
                        
                        for (std::size_t iKeypoint = 0; iKeypoint < kvt.second.size(); ++iKeypoint)
                        {
                            auto& keypoint = kvt.second[iKeypoint];
                            for (auto& keypointTrack : keypoint)
                            {
                                for (std::size_t iKeyFrame = 0; iKeyFrame < keypointTrack.size(); ++iKeyFrame)
                                {
                                    auto& keyframe = keypointTrack[iKeyFrame];
                                    keyframesFlat[keyframe.frameNumber][poseType][iPerson].push_back(
                                        KeyFrameFlat(keyframe.frameNumber, iPerson, iKeypoint,
                                            iKeyFrame == keypointTrack.size() - 1, keyframe.x, keyframe.y));
                                }
                            }
                        }
                    }

                    for (auto& track : person.headTranslationTracks)
                    {
                        for (std::size_t iKeyFrame = 0; iKeyFrame < track.size(); ++iKeyFrame)
                        {
                            auto& keyframe = track[iKeyFrame];
                            keyframes3DFlat[keyframe.frameNumber][0][iPerson].push_back(
                                KeyFrame3DFlat(keyframe.frameNumber, iPerson,
                                    iKeyFrame == track.size() - 1, keyframe.value));
                        }
                    }

                    for (auto& track : person.headRotationTracks)
                    {
                        for (std::size_t iKeyFrame = 0; iKeyFrame < track.size(); ++iKeyFrame)
                        {
                            auto& keyframe = track[iKeyFrame];
                            keyframes3DFlat[keyframe.frameNumber][1][iPerson].push_back(
                                KeyFrame3DFlat(keyframe.frameNumber, iPerson,
                                    iKeyFrame == track.size() - 1, keyframe.value));
                        }
                    }
                }

                for (std::size_t f = 0; f < mVideoNumFrames; ++f)
                {
                    if (f == 0 || (maxFramesPerSegment > 0 && f % maxFramesPerSegment == 0))
                    {
                        jRoots.push_back(nlohmann::json{});
                    }

                    nlohmann::json jFrame;

                    for (auto kv : keyframesFlat[f])
                    {
                        nlohmann::json jKeyframes;
                        auto poseType = kv.first;

                        for (auto& kvp : kv.second)
                        {
                            nlohmann::json jPerson;
                            auto personId = kvp.first;

                            for (auto& keyframe : kvp.second)
                            {
                                nlohmann::json jKeyframe;

                                //jKeyframe["n"] = keyframe.frameNumber;
                                //jKeyframe["p"] = keyframe.personId;
                                jKeyframe["k"] = keyframe.keypointId;
                                jKeyframe["l"] = int(keyframe.lastOne);
                                jKeyframe["x"] = int(round(keyframe.x));
                                jKeyframe["y"] = int(round(keyframe.y));

                                jPerson.push_back(jKeyframe);
                            }

                            if (!jPerson.empty())
                            {
                                jKeyframes[toString(personId)] = jPerson;
                            }
                        }

                        if (!jKeyframes.empty())
                        {
                            //nlohmann::json jPose;

                            std::string keypointName;
                            switch (poseType)
                            {
                            case 0:
                                keypointName = "pose";
                                break;
                            case 1:
                                keypointName = "face";
                                break;
                            case 2:
                                keypointName = "handL";
                                break;
                            case 3:
                                keypointName = "handR";
                                break;
                            default:
                                break;
                            }

                            //jPose["type"] = keypointName;
                            //jPose["keyframes"] = jKeyframes;
                            //jFrame.push_back(jPose);
                            jFrame[keypointName] = jKeyframes;
                        }
                    }

                    for (auto kv : keyframes3DFlat[f])
                    {
                        nlohmann::json jKeyframes;
                        auto poseType = kv.first;

                        for (auto& kvp : kv.second)
                        {
                            nlohmann::json jPerson;
                            auto personId = kvp.first;

                            for (auto& keyframe : kvp.second)
                            {
                                nlohmann::json jKeyframe;

                                //jKeyframe["n"] = keyframe.frameNumber;
                                //jKeyframe["p"] = keyframe.personId;
                                jKeyframe["l"] = int(keyframe.lastOne);
                                std::vector<int> value{
                                    int(round(keyframe.value.x * 1000)),
                                    int(round(keyframe.value.y * 1000)),
                                    int(round(keyframe.value.z * 1000))
                                };
                                jKeyframe["v"] = value;

                                jPerson.push_back(jKeyframe);
                            }

                            if (!jPerson.empty())
                            {
                                jKeyframes[toString(personId)] = jPerson;
                            }
                        }

                        if (!jKeyframes.empty())
                        {
                            //nlohmann::json jPose;

                            std::string keypointName;
                            switch (poseType)
                            {
                            case 0:
                                keypointName = "ht";
                                break;
                            case 1:
                                keypointName = "hr";
                                break;
                            default:
                                break;
                            }

                            //jPose["type"] = keypointName;
                            //jPose["keyframes"] = jKeyframes;
                            //jFrame.push_back(jPose);

                            jFrame[keypointName] = jKeyframes;
                        }
                    }

                    if (!jFrame.empty())
                    {
                        jRoots.back()["frames"][toString(f)] = jFrame;
                    }
                }

                for (std::size_t i = 0; i < jRoots.size(); ++i)
                {
                    auto& jRoot = jRoots[i];
                    jRoot["numSegments"] = jRoots.size();
                    jRoot["segmentIndex"] = i;
                    std::stringstream ss;
                    ss << this->mJsonPath << "_fbf_" << i;
                    std::ofstream ofile(ss.str() + ".json");
                    ofile << jRoot;
                    ofile.close();
#ifdef _WIN32
                    std::string command = "7z.exe a " + ss.str() + ".7z " + ss.str() + ".json";
                    system(command.c_str());
#endif
                }
            }

            if (outputMetafile)
            {
                // output video meta info
                nlohmann::json jMeta;
                jMeta["videoWidth"] = mVideoWidth;
                jMeta["videoHeight"] = mVideoHeight;
                jMeta["videoFPS"] = mVideoFPS;
                jMeta["videoNumFrames"] = mVideoNumFrames;
                jMeta["poseModel"] = getPoseModelName(mPoseModel);
                auto poseBodyPartMapping = getPoseBodyPartMapping(mPoseModel);
                for (auto kv : poseBodyPartMapping)
                {
                    jMeta["keypoints"][toString(kv.first)] = kv.second;
                }

                auto poseBodyHierarchy = getPoseBodyHierarchy(mPoseModel);
                for (const auto& v : poseBodyHierarchy)
                {
                    std::vector<unsigned int> temp(v.begin() + 1, v.end());
                    jMeta["hierarchy"][toString(v.front())] = temp;
                }

                std::ofstream ofile(this->mJsonPath + "_meta.json");
                ofile << jMeta;
                ofile.close();
            }
        }
    };
}


// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    COMPILE_TEMPLATE_DATUM(WJsonOutput);
}

#endif