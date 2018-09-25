#ifndef OPENPOSE_POSE_POSE_CPU_RENDERER_HPP
#define OPENPOSE_POSE_POSE_CPU_RENDERER_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/renderer.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/pose/poseParametersRender.hpp>
#include <openpose/pose/poseRenderer.hpp>
#include <nlohmann/json.hpp>
#include <sstream>
#include <fstream>

using namespace nlohmann;

namespace op
{
    struct PoseTrackingInfo
    {
        struct Person
        {
            std::size_t startFrame;
            std::size_t endFrame;
            int id;

            struct Track
            {
                std::string type;

                struct Keypoint
                {
                    struct KeyframeClip
                    {
                        struct Keyframe
                        {
                            std::size_t frameNumber;
                            float x;
                            float y;

                            Keyframe(std::size_t _frameNumber, float _x, float _y)
                                : frameNumber(_frameNumber)
                                , x(_x)
                                , y(_y)
                            {
                            }
                        };
                        std::vector<std::shared_ptr<Keyframe>> keyframes;

                        KeyframeClip(const json& j)
                        {
                            for (auto keyframe : j)
                            {
                                keyframes.push_back(std::make_shared<Keyframe>(keyframe["n"], keyframe["x"], keyframe["y"]));
                            }
                        }
                    };
                    std::vector<std::shared_ptr<KeyframeClip>> keyframeClips;

                    explicit Keypoint(const json& j)
                    {
                        for (auto keyframeClip : j)
                        {
                            keyframeClips.push_back(std::make_shared<KeyframeClip>(keyframeClip));
                        }
                    }
                };

                std::map<std::size_t, std::shared_ptr<Keypoint>> keypoints;

                explicit Track(const std::string& type_, const json& jTrack)
                    : type{ type_ }
                {
                    for (auto iKeypoint = jTrack.begin(); iKeypoint != jTrack.end(); ++iKeypoint)
                    {
                        std::stringstream ss;
                        ss << iKeypoint.key();
                        std::size_t key = 0;
                        ss >> key;
                        keypoints[key] = std::make_shared<Keypoint>(iKeypoint.value());
                    }
                }
            };
            std::vector<std::shared_ptr<Track>> tracks;

            struct Keyframe3D
            {
                std::size_t frameNumber;
                cv::Point3d value;

                explicit Keyframe3D(const json& j)
                    : frameNumber(j["n"])
                {
                    std::vector<int> v = j["v"];
                    value.x = v[0] / 1000.0;
                    value.y = v[1] / 1000.0;
                    value.z = v[2] / 1000.0;
                }
            };
            std::vector<std::vector<std::shared_ptr<Keyframe3D>>> faceRotationTracks;
            std::vector<std::vector<std::shared_ptr<Keyframe3D>>> faceTranslationTracks;

            explicit Person(const json& j)
            {
                startFrame = j["startFrame"];
                endFrame = j["endFrame"];
                id = j["id"];

                auto jTracks = j["tracks"];

                for (auto iTrack = jTracks.begin(); iTrack != jTracks.end(); ++iTrack)
                {
                    if (iTrack.key() == "head")
                    {
                        for (auto track : iTrack.value()["r"])
                        {
                            faceRotationTracks.push_back(std::vector<std::shared_ptr<Keyframe3D>>{});

                            for (auto keyframe : track)
                            {
                                faceRotationTracks.back().push_back(std::make_shared<Keyframe3D>(keyframe));
                            }
                        }

                        for (auto track : iTrack.value()["t"])
                        {
                            faceTranslationTracks.push_back(std::vector<std::shared_ptr<Keyframe3D>>{});

                            for (auto keyframe : track)
                            {
                                faceTranslationTracks.back().push_back(std::make_shared<Keyframe3D>(keyframe));
                            }
                        }
                    }
                    else
                    {
                        tracks.push_back(std::make_shared<Track>(iTrack.key(), iTrack.value()));
                    }
                }
            }
        };

        std::vector<std::shared_ptr<Person>> persons;
        explicit PoseTrackingInfo(const std::string& poseTrackingInfoFolder)
        {
            int numSegments = 1;
            for (int s = 0; s < numSegments; ++s)
            {
                std::stringstream ss;
                ss << poseTrackingInfoFolder << "pbp_" << s << ".json";

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
    };

    class OP_API PoseCpuRenderer : public Renderer, public PoseRenderer
    {
    public:
        PoseCpuRenderer(const PoseModel poseModel, const float renderThreshold, const bool blendOriginalFrame = true,
                        const float alphaKeypoint = POSE_DEFAULT_ALPHA_KEYPOINT,
                        const float alphaHeatMap = POSE_DEFAULT_ALPHA_HEAT_MAP,
                        const unsigned int elementToRender = 0u);

        std::pair<int, std::string> renderPose(Array<float>& outputData,
            const Array<float>& poseKeypoints,
            const std::size_t frameNumber,
            const float scaleInputToOutput,
            const float scaleNetToOutput = -1.f);
    private:
        DELETE_COPY(PoseCpuRenderer);
    };

    class OP_API PoseTrackingInfoVisualizer : public Renderer, public PoseRenderer
    {
    public:
        PoseTrackingInfoVisualizer(const PoseModel poseModel, const std::string& poseTrackingInfoFolder = "",
            const float renderThreshold = 0.f, const bool blendOriginalFrame = true,
            const float alphaKeypoint = POSE_DEFAULT_ALPHA_KEYPOINT,
            const float alphaHeatMap = POSE_DEFAULT_ALPHA_HEAT_MAP,
            const unsigned int elementToRender = 0u);

        std::pair<int, std::string> renderPose(
            Array<float>& outputData,
            const Array<float>& poseKeypoints,
            const std::size_t frameNumber,
            const float scaleInputToOutput,
            const float scaleNetToOutput = -1.f);

        std::shared_ptr<PoseTrackingInfo> getPoseTrackingInfo()
        {
            return mPoseTrackingInfo;
        }
    private:
        DELETE_COPY(PoseTrackingInfoVisualizer);

        void renderPoseTrackingInfo(Array<float>& outputData, const std::size_t frameNumber);

        template <typename T1, typename T2>
        T1 lerp(const T1& p1, const T1& p2, T2 frameNumber1, T2 frameNumber2, T2 frameNumberCurrent)
        {
            auto t = float(frameNumberCurrent - frameNumber1) / float(frameNumber2 - frameNumber1);

            return p1 - p1 * t + p2 * t;
        }

        std::shared_ptr<PoseTrackingInfo> mPoseTrackingInfo;
    };
}

#endif // OPENPOSE_POSE_POSE_CPU_RENDERER_HPP
