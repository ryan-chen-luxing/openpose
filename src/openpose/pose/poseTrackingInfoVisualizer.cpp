#include <openpose/pose/renderPose.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/pose/poseCpuRenderer.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <tuple>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <sstream>

using namespace nlohmann;

namespace op
{
    PoseTrackingInfo::PoseTrackingInfo(const std::string& fileid)
    {
        int numSegments = 1;
        for (int s = 0; s < numSegments; ++s)
        {
            std::stringstream ss;
            ss << fileid << "_" << s << ".json";

            std::ifstream i(ss.str());
            json j;
            i >> j;

            numSegments = j["numSegments"];

            for (auto person : j["persons"])
            {
                persons.push_back(std::make_shared<Person>(person));
            }
        }
    }

    PoseTrackingInfoVisualizer::PoseTrackingInfoVisualizer(const PoseModel poseModel,
        const std::string& poseTrackingInfo,
        const float renderThreshold, const bool blendOriginalFrame,
        const float alphaKeypoint,
        const float alphaHeatMap,
        const unsigned int elementToRender)
        : Renderer { renderThreshold, alphaKeypoint,
        alphaHeatMap, blendOriginalFrame, elementToRender,
        getNumberElementsToRender(poseModel)}
        , PoseRenderer{ poseModel }
    {
        if (!poseTrackingInfo.empty())
        {
            mPoseTrackingInfo = std::make_shared<PoseTrackingInfo>(poseTrackingInfo);
        }
    }

    std::pair<int, std::string> PoseTrackingInfoVisualizer::renderPose(
        Array<float>& outputData,
        const Array<float>& poseKeypoints,
        const std::size_t frameNumber,
        const float scaleInputToOutput,
        const float scaleNetToOutput)
    {
        try
        {
            // Security checks
            if (outputData.empty())
                error("Empty Array<float> outputData.", __LINE__, __FUNCTION__, __FILE__);
            // CPU rendering
            const auto elementRendered = spElementToRender->load();
            std::string elementRenderedName;
            // Draw poseKeypoints
            if (elementRendered == 0)
            {
                renderPoseTrackingInfo(outputData, frameNumber);

                // Rescale keypoints to output size
                //auto poseKeypointsRescaled = poseKeypoints.clone();
                //scaleKeypoints(poseKeypointsRescaled, scaleInputToOutput);
                // Render keypoints
                //renderPoseKeypointsCpu(outputData, poseKeypointsRescaled, mPoseModel, mRenderThreshold,
                //    frameNumber, mBlendOriginalFrame);
            }
            // Draw heat maps / PAFs
            else
            {
                UNUSED(scaleNetToOutput);
                error("CPU rendering only available for drawing keypoints, no heat maps nor PAFs.",
                    __LINE__, __FUNCTION__, __FILE__);
            }
            // Return result
            return std::make_pair(elementRendered, elementRenderedName);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_pair(-1, "");
        }
    }

    void PoseTrackingInfoVisualizer::renderPoseTrackingInfo(Array<float>& outputData, const std::size_t frameNumber)
    {
        if (mPoseTrackingInfo)
        {
            auto frame = outputData.getCvMat();

            typedef std::vector<cv::Point2f> Keypoints;
            typedef std::vector<Keypoints> Tracks;
            typedef std::tuple<int, Tracks, cv::Mat, cv::Mat> Person;
            typedef std::vector<Person> Persons;

            Persons persons;
            for (auto person : mPoseTrackingInfo->persons)
            {
                if (person->startFrame <= frameNumber && person->endFrame >= frameNumber)
                {
                    Tracks tracks;

                    for (auto track : person->tracks)
                    {
                        std::vector<cv::Point2f> keypoints;

                        for (auto kvkp : track->keypoints.keypoints)
                        {
                            cv::Point2f keypoint(-1.f, -1.f);

                            auto itrKeyframeClips = kvkp.second->keyframeClips.begin();
                            while (itrKeyframeClips != kvkp.second->keyframeClips.end())
                            {
                                if (!(*itrKeyframeClips)->keyframes.empty() &&
                                    (*itrKeyframeClips)->keyframes.front()->frameNumber <= frameNumber &&
                                    (*itrKeyframeClips)->keyframes.back()->frameNumber >= frameNumber)
                                {
                                    break;
                                }

                                ++itrKeyframeClips;
                            }

                            if (itrKeyframeClips != kvkp.second->keyframeClips.end())
                            {
                                auto itr = (*itrKeyframeClips)->keyframes.begin();
                                while (itr != (*itrKeyframeClips)->keyframes.end())
                                {
                                    if (itr == (*itrKeyframeClips)->keyframes.begin())
                                    {
                                        if ((*itr)->frameNumber > frameNumber)
                                            // haven't reached the keyframes yet
                                            break;
                                    }
                                    else if ((*itr)->frameNumber >= frameNumber)
                                    {
                                        break;
                                    }

                                    ++itr;
                                }

                                if (itr != (*itrKeyframeClips)->keyframes.begin() &&
                                    itr != (*itrKeyframeClips)->keyframes.end())
                                {
                                    auto previousItr = itr - 1;
                                    keypoint = lerp(
                                        cv::Point2f{ (*previousItr)->x, (*previousItr)->y },
                                        cv::Point2f{ (*itr)->x, (*itr)->y },
                                        (*previousItr)->frameNumber,
                                        (*itr)->frameNumber, frameNumber);
                                }
                            }

                            keypoints.push_back(keypoint);
                        }

                        tracks.push_back(keypoints);
                    }

                    cv::Mat translation;

                    for (auto track : person->faceTranslationTracks)
                    {
                        if (!track.empty() &&
                            frameNumber >= track.front()->frameNumber &&
                            frameNumber <= track.back()->frameNumber)
                        {
                            for (auto f = track.begin(); f != track.end(); ++f)
                            {
                                if ((*f)->frameNumber >= frameNumber && f != track.begin())
                                {
                                    auto lerpValue = lerp(
                                        (*(f - 1))->value, (*f)->value,
                                        (*(f - 1))->frameNumber,
                                        (*f)->frameNumber, frameNumber);

                                    translation = (cv::Mat_<double>(3, 1) << lerpValue.x, lerpValue.y, lerpValue.z);

                                    break;
                                }
                            }
                            break;
                        }
                    }

                    cv::Mat rotation;

                    for (auto track : person->faceRotationTracks)
                    {
                        if (!track.empty() &&
                            frameNumber >= track.front()->frameNumber &&
                            frameNumber <= track.back()->frameNumber)
                        {
                            for (auto f = track.begin(); f != track.end(); ++f)
                            {
                                if ((*f)->frameNumber >= frameNumber &&
                                    f != track.begin())
                                {
                                    auto lerpValue = lerp(
                                        (*(f - 1))->value, (*f)->value,
                                        (*(f - 1))->frameNumber,
                                        (*f)->frameNumber, frameNumber);

                                    rotation = (cv::Mat_<double>(3, 1) << lerpValue.x, lerpValue.y, lerpValue.z);

                                    break;
                                }
                            }
                            break;
                        }
                    }

                    persons.push_back(std::make_tuple(person->id, tracks, translation, rotation));
                }
            }

            if (!persons.empty() && !std::get<1>(persons.front()).empty())
            {
                // Parameters
                const auto thicknessCircleRatio = 1.f / 75.f;
                const auto thicknessLineRatioWRTCircle = 0.75f;
                const auto& pairs = getPoseBodyPartPairsRender(mPoseModel);
                const auto& poseScales = getPoseScales(mPoseModel);
                const auto& colors = getPoseColors(mPoseModel);

                // Get frame channels
                const auto width = frame.size[1];
                const auto height = frame.size[0];
                const auto area = width * height;
                cv::Mat frameBGR(height, width, CV_32FC3, frame.data);

                // Parameters
                const auto lineType = 8;
                const auto shift = 0;
                const auto numberColors = colors.size();
                const auto numberScales = poseScales.size();
                const auto thresholdRectangle = 0.1f;
                const auto numberKeypoints = std::get<1>(persons.front()).front().size();

                const auto threshold = 0.5f;

                // Keypoints
                for (const auto& person : persons)
                {
                    Rectangle<int> personRectangle{};
                    {
                        // Security checks
                        if (numberKeypoints < 1)
                            error("Number body parts must be > 0", __LINE__, __FUNCTION__, __FILE__);
                        // Define keypointPtr
                        int minX = std::numeric_limits<int>::max();
                        int maxX = 0;
                        int minY = minX;
                        int maxY = maxX;
                        for (const auto& keypoint : std::get<1>(person).front())
                        {
                            const auto x = keypoint.x;
                            const auto y = keypoint.y;
                            if (x < 0 || y < 0)
                                continue;

                            // Set X
                            if (maxX < x)
                                maxX = x;
                            if (minX > x)
                                minX = x;
                            // Set Y
                            if (maxY < y)
                                maxY = y;
                            if (minY > y)
                                minY = y;
                        }
                        if (maxX >= minX && maxY >= minY)
                            personRectangle = Rectangle<int>{ minX, minY, maxX - minX, maxY - minY };
                    }

                    if (personRectangle.area() > 0)
                    {
                        const auto ratioAreas = fastMin(1.f, fastMax(personRectangle.width / (float)width,
                            personRectangle.height / (float)height));
                        // Size-dependent variables
                        const auto thicknessRatio = fastMax(intRound(std::sqrt(area)
                            * thicknessCircleRatio * ratioAreas), 2);
                        // Negative thickness in cv::circle means that a filled circle is to be drawn.
                        const auto thicknessCircle = fastMax(1, (ratioAreas > 0.05f ? thicknessRatio : -1));
                        const auto thicknessLine = fastMax(1, intRound(thicknessRatio * thicknessLineRatioWRTCircle));
                        const auto radius = thicknessRatio / 2;

                        // Draw lines
                        for (auto iPair = 0u; iPair < pairs.size(); iPair += 2)
                        {
                            std::pair<unsigned int, unsigned int> pair{ pairs[iPair], pairs[iPair + 1] };

                            const auto& part1 = std::get<1>(person).front()[pair.first];
                            const auto& part2 = std::get<1>(person).front()[pair.second];

                            if (part1.x > 0 && part1.y > 0 && part2.x > 0 && part2.y > 0)
                            {
                                const auto thicknessLineScaled = thicknessLine
                                    * poseScales[pair.second % numberScales];
                                const auto colorIndex = pair.second * 3; // Before: colorIndex = pair/2*3;
                                const cv::Scalar color{
                                    colors[(colorIndex + 2) % numberColors],
                                    colors[(colorIndex + 1) % numberColors],
                                    colors[colorIndex % numberColors]
                                };
                                const cv::Point keypoint1{ int(round(part1.x)), int(round(part1.y)) };
                                const cv::Point keypoint2{ int(round(part2.x)), int(round(part2.y)) };
                                cv::line(frameBGR, keypoint1, keypoint2, color, thicknessLineScaled, lineType, shift);
                            }
                        }

                        // Draw circles
                        for (int part = 0; part < std::get<1>(person).front().size(); ++part)
                        {
                            const auto& keypoint = std::get<1>(person).front()[part];
                            if (keypoint.x > 0 && keypoint.y > 0)
                            {
                                const auto radiusScaled = radius * poseScales[part % numberScales];
                                const auto thicknessCircleScaled = thicknessCircle * poseScales[part % numberScales];
                                const auto colorIndex = part * 3;
                                const cv::Scalar color{
                                    colors[(colorIndex + 2) % numberColors],
                                    colors[(colorIndex + 1) % numberColors],
                                    colors[colorIndex % numberColors]
                                };
                                const cv::Point center{ int(round(keypoint.x)), int(round(keypoint.y)) };
                                cv::circle(frameBGR, center, radiusScaled, color, thicknessCircleScaled, lineType, shift);
                            }
                        }

                        const auto& origin = std::get<1>(person).front().front();
                        if (origin.x > 0 && origin.y > 0)
                        {
                            const auto fontScale = fastMax(1, fastMin(2, int(round(float(width) / 480))));
                            cv::putText(frameBGR, std::to_string(std::get<0>(person)),
                            { int(round(origin.x)), int(round(origin.y + 10)) },
                                0, fontScale, { 255.f, 0.f, 0.f }, 2);
                        }

                        const auto translation = std::get<2>(person);
                        const auto rotation = std::get<3>(person);
                        if (translation.rows > 0 && rotation.rows > 0)
                        {
                            double focal_length = frameBGR.cols; // Approximate focal length.
                            cv::Point2d center = cv::Point2d(frameBGR.cols / 2, frameBGR.rows / 2);
                            cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
                            cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion

                            std::vector<cv::Point3d> nose_end_point3D;
                            std::vector<cv::Point2d> nose_end_point2D;
                            nose_end_point3D.push_back(cv::Point3d(0, 0, 2.0));

                            cv::projectPoints(nose_end_point3D, rotation, translation, camera_matrix, dist_coeffs, nose_end_point2D);

                            cv::Point2d nosePoint(std::get<1>(person).front()[0].x, std::get<1>(person).front()[0].y);
                            cv::line(frameBGR, nosePoint, nose_end_point2D[0], cv::Scalar(255, 0, 0), 2);
                        }
                    }
                }
            }
        }
    }
}
