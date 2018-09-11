#ifndef OPENPOSE_POSE_POSE_CPU_RENDERER_HPP
#define OPENPOSE_POSE_POSE_CPU_RENDERER_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/renderer.hpp>
#include <openpose/pose/enumClasses.hpp>
#include <openpose/pose/poseParametersRender.hpp>
#include <openpose/pose/poseRenderer.hpp>
#include <nlohmann/json.hpp>

using namespace nlohmann;

namespace op
{
    struct PoseTrackingInfo
    {
        struct Person
        {
            int startFrame;
            int endFrame;
            int id;

            struct Track
            {
                std::string type;

                struct Keypoints
                {
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

                    explicit Keypoints(const json& j)
                    {
                        for (std::size_t i = 0; i < j.size(); ++i)
                        {
                            keypoints[i] = std::make_shared<Keypoint>(j[i]);
                        }
                    }
                };
                Keypoints keypoints;

                explicit Track(const json& j)
                    : keypoints(j["keypoints"])
                {
                    type = j["type"];
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

                for (auto track : j["tracks"])
                {
                    tracks.push_back(std::make_shared<Track>(track));
                }

                for (auto track : j["faceTranslation"])
                {
                    faceTranslationTracks.push_back(std::vector<std::shared_ptr<Keyframe3D>>{});

                    for (auto keyframe : track)
                    {
                        faceTranslationTracks.back().push_back(std::make_shared<Keyframe3D>(keyframe));
                    }
                }

                for (auto track : j["faceRotation"])
                {
                    faceRotationTracks.push_back(std::vector<std::shared_ptr<Keyframe3D>>{});

                    for (auto keyframe : track)
                    {
                        faceRotationTracks.back().push_back(std::make_shared<Keyframe3D>(keyframe));
                    }
                }
            }
        };

        std::vector<std::shared_ptr<Person>> persons;
        explicit PoseTrackingInfo(const std::string& fileid);
    };

    class OP_API PoseCpuRenderer : public Renderer, public PoseRenderer
    {
    public:
        PoseCpuRenderer(const PoseModel poseModel, const float renderThreshold, const bool blendOriginalFrame = true,
                        const float alphaKeypoint = POSE_DEFAULT_ALPHA_KEYPOINT,
                        const float alphaHeatMap = POSE_DEFAULT_ALPHA_HEAT_MAP,
                        const unsigned int elementToRender = 0u);

        std::pair<int, std::string> renderPose(Array<float>& outputData, const Array<float>& poseKeypoints,
                                               const unsigned long long frameNumber,
                                               const float scaleInputToOutput,
                                               const float scaleNetToOutput = -1.f);
    private:
        DELETE_COPY(PoseCpuRenderer);
    };

    class OP_API PoseTrackingInfoVisualizer : public Renderer, public PoseRenderer
    {
    public:
        PoseTrackingInfoVisualizer(const PoseModel poseModel, const std::string& poseTrackingInfo = "",
            const float renderThreshold = 0.f, const bool blendOriginalFrame = true,
            const float alphaKeypoint = POSE_DEFAULT_ALPHA_KEYPOINT,
            const float alphaHeatMap = POSE_DEFAULT_ALPHA_HEAT_MAP,
            const unsigned int elementToRender = 0u);

        std::pair<int, std::string> renderPose(
            Array<float>& outputData,
            const Array<float>& poseKeypoints,
            const unsigned long long frameNumber,
            const float scaleInputToOutput,
            const float scaleNetToOutput = -1.f);

        std::shared_ptr<PoseTrackingInfo> getPoseTrackingInfo()
        {
            return mPoseTrackingInfo;
        }
    private:
        DELETE_COPY(PoseTrackingInfoVisualizer);

        void renderPoseTrackingInfo(Array<float>& outputData, const unsigned long long frameNumber);

        template <typename T1, typename T2>
        T1 lerp(const T1& p1, const T1& p2, T2 frameNumber1, T2 frameNumber2, T2 frameNumberCurrent)
        {
            float t = float(frameNumberCurrent - frameNumber1) / float(frameNumber2 - frameNumber1);

            return p1 * (1 - t) + p2 * t;
        }

        std::shared_ptr<PoseTrackingInfo> mPoseTrackingInfo;
    };
}

#endif // OPENPOSE_POSE_POSE_CPU_RENDERER_HPP
