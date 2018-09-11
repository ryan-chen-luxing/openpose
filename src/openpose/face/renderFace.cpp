#include <openpose/face/faceParameters.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/face/renderFace.hpp>
#include <opencv2/opencv.hpp>

namespace op
{
    std::vector<float> averageConfidence(const Array<float>& keypoints, const std::vector<unsigned int> keypointsIndices)
    {
        std::vector<float> result;

        const auto numberKeypoints = keypoints.getSize(1);
        // Keypoints
        for (std::size_t person = 0; person < keypoints.getSize(0); person++)
        {
            float average = 0.f;

            // Draw circles
            for (std::size_t i = 0; i < keypointsIndices.size(); i++)
            {
                auto part = keypointsIndices[i];
                const auto faceIndex = (person * numberKeypoints + part) * keypoints.getSize(2);
                const auto confidence = keypoints[faceIndex + 2];
                average += confidence;
            }

            average /= float(keypointsIndices.size());

            result.push_back(average);
        }

        return result;
    }

    void renderFaceKeypointsCpu(Array<float>& frameArray, const Array<float>& faceKeypoints,
                                const float renderThreshold)
    {
        try
        {
            if (!frameArray.empty())
            {
                // Parameters
                const auto thicknessCircleRatio = 1.f/75.f;
                const auto thicknessLineRatioWRTCircle = 0.334f;
                const auto& pairs = FACE_PAIRS_RENDER;
                const auto& scales = FACE_SCALES_RENDER;

                // Render keypoints
                renderKeypointsCpu(frameArray, faceKeypoints, pairs, FACE_COLORS_RENDER, thicknessCircleRatio,
                                   thicknessLineRatioWRTCircle, scales, renderThreshold);
                /*
                if (!frameArray.empty())
                {
                    // Array<float> --> cv::Mat
                    auto frame = frameArray.getCvMat();
                    const auto width = frame.size[1];
                    const auto height = frame.size[0];
                    cv::Mat frameBGR(height, width, CV_32FC3, frame.data);

                    // Security check
                    if (frame.channels() != 3)
                        error("The Array<float> is not a RGB image or 3-channel keypoint array. This function"
                            " is only for array of dimension: [sizeA x sizeB x 3]."
                            , __LINE__, __FUNCTION__, __FILE__);

                    // Keypoints
                    for (auto person = 0; person < faceKeypoints.getSize(0); person++)
                    {
                        const std::vector<unsigned int> faceKeypointsPoseEstimationIndices
                        {
                            33, 8, 36, 45, 48, 54
                        };

                        const auto numberKeypoints = faceKeypoints.getSize(1);
                        // Keypoints

                        // 2D image points.
                        std::vector<cv::Point2d> image_points;
                        for (auto i = 0; i < faceKeypointsPoseEstimationIndices.size(); i++)
                        {
                            auto part = faceKeypointsPoseEstimationIndices[i];
                            const auto faceIndex = (person * numberKeypoints + part) * faceKeypoints.getSize(2);
                            image_points.push_back(cv::Point2d(faceKeypoints[faceIndex], faceKeypoints[faceIndex + 1]));
                        }

                        // 3D model points.
                        std::vector<cv::Point3d> model_points;
                        model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // Nose tip
                        model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));          // Chin
                        model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));       // Left eye left corner
                        model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));        // Right eye right corner
                        model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));      // Left Mouth corner
                        model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));       // Right mouth corner


                                                                                                // Camera internals
                        double focal_length = frameBGR.cols; // Approximate focal length.
                        cv::Point2d center = cv::Point2d(frameBGR.cols / 2, frameBGR.rows / 2);
                        cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
                        cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion

                                                                                                // Output rotation and translation
                        cv::Mat rotation_vector; // Rotation in axis-angle form
                        cv::Mat translation_vector;

                        // Solve for pose
                        cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

                        // Project a 3D point (0, 0, 1000.0) onto the image plane.
                        // We use this to draw a line sticking out of the nose

                        std::vector<cv::Point3d> nose_end_point3D;
                        std::vector<cv::Point2d> nose_end_point2D;
                        nose_end_point3D.push_back(cv::Point3d(0, 0, 1000.0));

                        cv::projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);

                        for (int i = 0; i < image_points.size(); i++)
                        {
                            cv::circle(frameBGR, image_points[i], 3, cv::Scalar(0, 0, 255), -1);
                        }

                        cv::line(frameBGR, image_points[0], nose_end_point2D[0], cv::Scalar(255, 0, 0), 2);

                    }
                }
                */
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
