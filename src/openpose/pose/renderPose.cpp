#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/pose/renderPose.hpp>
#include <opencv2/opencv.hpp>

namespace op
{
    void renderPoseKeypointsCpu(Array<float>& frameArray, const Array<float>& poseKeypoints, const PoseModel poseModel,
                                const float renderThreshold, const std::size_t frameNumber, const bool blendOriginalFrame)
    {
        try
        {
            if (!frameArray.empty())
            {
                // Background
                if (!blendOriginalFrame)
                    frameArray.getCvMat().setTo(0.f); // [0-255]

                // Parameters
                const auto thicknessCircleRatio = 1.f/75.f;
                const auto thicknessLineRatioWRTCircle = 0.75f;
                const auto& pairs = getPoseBodyPartPairsRender(poseModel);
                const auto& poseScales = getPoseScales(poseModel);

                // Render keypoints
                renderKeypointsCpu(frameArray, poseKeypoints, pairs, getPoseColors(poseModel), thicknessCircleRatio,
                                   thicknessLineRatioWRTCircle, poseScales, renderThreshold);

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
                    for (auto person = 0; person < poseKeypoints.getSize(0); person++)
                    {
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
                        if (poseModel == PoseModel::BODY_25)
                        {
                            keypointsPoseEstimationIndices = keypointsPoseEstimationIndicesBody25;
                        }
                        else if (poseModel == PoseModel::COCO_18)
                        {
                            keypointsPoseEstimationIndices = keypointsPoseEstimationIndicesCoco;
                        }

                        /*
                        // 3D model points.
                        const std::vector<cv::Point3d> keypointsReferencePose
                        {
                            cv::Point3d(0.0f, 0.0f, 0.0f),
                            cv::Point3d(-20.0f, 17.0f, -13.5f),
                            cv::Point3d(20.0f, 17.0f, -13.5f),
                            cv::Point3d(-40.0f, 12.0f, -70.0f),
                            cv::Point3d(40.0f, 12.0f, -70.0f)
                        };
                        */
                        // 3D model points.
                        const std::vector<cv::Point3d> keypointsReferencePose
                        {
                            cv::Point3d(0, 0, 0),
                            cv::Point3d(-224, 209, -261),
                            cv::Point3d(224, 209, -261),
                            cv::Point3d(-644, 130, -1000),
                            cv::Point3d(644, 130, -1000)
                        };

                        const auto numberKeypoints = poseKeypoints.getSize(1);
                        // Keypoints

                        // 2D image points.
                        std::vector<cv::Point2d> image_points;
                        std::vector<cv::Point3d> model_points;
                        for (std::size_t i = 0; i < keypointsPoseEstimationIndices.size(); i++)
                        {
                            auto part = keypointsPoseEstimationIndices[i];
                            const auto partIndex = (person * numberKeypoints + part) * poseKeypoints.getSize(2);

                            auto confidence = poseKeypoints[partIndex + 2];

                            if (confidence > renderThreshold)
                            {
                                image_points.push_back(cv::Point2d(poseKeypoints[partIndex], poseKeypoints[partIndex + 1]));
                                model_points.push_back(keypointsReferencePose[i]);
                            }
                        }

                        if (image_points.size() >= 4)
                        {
                            // Camera internals
                            double focal_length = frameBGR.cols; // Approximate focal length.
                            cv::Point2d center = cv::Point2d(frameBGR.cols / 2, frameBGR.rows / 2);
                            cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
                            cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion

                                                                                                    // Output rotation and translation
                            cv::Mat rotation; // Rotation in axis-angle form
                            cv::Mat translation;

                            // Solve for pose
                            cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation, translation, false, cv::SOLVEPNP_ITERATIVE);

                            //cv::solvePnPRansac(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector, false, 100, 4.f);

                            // Project a 3D point (0, 0, 1000.0) onto the image plane.
                            // We use this to draw a line sticking out of the nose

                            std::vector<cv::Point3d> nose_end_point3D;
                            std::vector<cv::Point2d> nose_end_point2D;
                            nose_end_point3D.push_back(cv::Point3d(0, 0, 1000.0));

                            //cv::Mat J;
                            cv::projectPoints(nose_end_point3D, rotation, translation, camera_matrix, dist_coeffs, nose_end_point2D);
                            //std::cout << std::endl << "Jacobian Matrix:" << std::endl << J;
                            cv::line(frameBGR, image_points[0], nose_end_point2D[0], cv::Scalar(255, 0, 0), 2);

                            //cv::Mat Sigma = cv::Mat(J.t() * J, cv::Rect(0, 0, 6, 6)).inv();

                            // Compute standard deviation
                            //cv::Mat std_dev;
                            //sqrt(Sigma.diag(), std_dev);
                            //std::cout << std::endl << "rvec1, tvec1 standard deviation:" << std::endl << std_dev;
                        }
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
