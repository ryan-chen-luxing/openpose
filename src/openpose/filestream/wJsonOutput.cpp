#include <openpose/filestream/wJsonOutput.hpp>
#include <algorithm>
#include <limits>
#include <sstream>

namespace op
{
template<typename TDatums>
void WJsonOutput<TDatums>::postProcess(float confidenceThreshold, float mseLerpTheshold,
    double mseHeadTranslationThreshold, double mseHeadRotationThreshold,
    int indexerInterval, int maxFramesPerSegment)
{
    // processing raw frames into skeletons
    std::vector<PersonInfo> personsRaw;
    std::vector<unsigned int> personsBeingTracked;
    for (const auto& rawframe : this->mRawFrames)
    {
        auto numberPeople = 0;
        for (auto vectorIndex = 0u; vectorIndex < rawframe.keypoints.size(); vectorIndex++)
        {
            numberPeople = fastMax(numberPeople, rawframe.keypoints[vectorIndex].getSize(0));
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

        std::vector<unsigned int> personsMatched;
        std::vector<unsigned int> newPersonsIdentified;
        for (auto personIndex = 0; personIndex < numberPeople; personIndex++)
        {
            const auto personRectangle = getKeypointsRectangle(rawframe.keypoints[0], personIndex, confidenceThreshold);
            if (personRectangle.area() > 0)
            {
                std::vector<std::vector<FrameRaw>> rawFrames;
                for (auto vectorIndex = 0u; vectorIndex < rawframe.keypoints.size(); vectorIndex++)
                {
                    const auto& keypoints = rawframe.keypoints[vectorIndex];

                    const auto numberElementsPerRaw = keypoints.getSize(1) * keypoints.getSize(2);

                    std::vector<FrameRaw> frames;

                    if (numberElementsPerRaw > 0)
                    {
                        const auto finalIndex = personIndex * numberElementsPerRaw;

                        for (auto part = 0; part < keypoints.getSize(1); part++)
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
                    unsigned int personBestMatched = -1;
                    for (auto i : personsBeingTracked)
                    {
                        // check if the person has been matched already in the personsMatched array.
                        if (std::find(personsMatched.begin(), personsMatched.end(), i) != personsMatched.end())
                            continue;

                        float sumDifference = 0.f;

                        const auto& personTracked = personsRaw[i];

                        const auto& rawFramesTracked = personTracked.frames.at(rawframe.frameNumber - 1);

                        auto numValidParts = 0;
                        for (auto trackingType = 0; trackingType < rawFrames.size(); ++trackingType)
                        {
                            for (auto part = 0; part < rawFrames[trackingType].size(); ++part)
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

        personsBeingTracked.erase(std::remove_if(personsBeingTracked.begin(), personsBeingTracked.end(), [&](unsigned int index)
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
                    auto areaOverlappingPercentageBest = 0.6f;
                    auto personBestMatched = (*classificationItr)->objectsInfo.end();

                    for (auto objectInfoItr = (*classificationItr)->objectsInfo.begin();
                        objectInfoItr != (*classificationItr)->objectsInfo.end(); ++objectInfoItr)
                    {
                        const auto x1 = (*objectInfoItr)->x1 * this->mVideoWidth;
                        const auto y1 = (*objectInfoItr)->y1 * this->mVideoHeight;
                        const auto x2 = (*objectInfoItr)->x2 * this->mVideoWidth;
                        const auto y2 = (*objectInfoItr)->y2 * this->mVideoHeight;

                        const std::tuple<float, float, float, float> identityRect{ x1, y1, x2, y2 };
                        const auto areaIdentity = rectArea(identityRect);


                        auto rectOverlapping = overlappingRect(personRect, identityRect);

                        auto areaPerson = rectArea(personRect);
                        auto areaOverlapping = rectArea(rectOverlapping);

                        float areaOverlappingPercentageAvg = (areaOverlapping / areaPerson + areaOverlapping / areaIdentity) * 0.5;

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
    const std::vector<unsigned int> keypointsPoseEstimationIndices
    {
        0, 15, 16, 17, 18
    };

    // 3D model points.
    const std::vector<cv::Point3d> keypointsReferencePose
    {
        cv::Point3d(0, 0, 0),
        cv::Point3d(-0.224, 0.209, -0.261),
        cv::Point3d(0.224, 0.209, -0.261),
        cv::Point3d(-0.644, 0.130, -1),
        cv::Point3d(0.644, 0.130, -1)
    };

    const std::vector<unsigned int> keypointsFaceEstimationIndices
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
            for (auto i = 0; i < keypointsPoseEstimationIndices.size(); i++)
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
                for (auto i = 0; i < keypointsFaceEstimationIndices.size(); i++)
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

    // Moving average based transform filtering
    //for (auto& p : persons)
    //{
    //    auto& person = *p;

    //    const int windowSize = 30;
    //    const unsigned int stepSize = 1;
    //    const float deviationThreshold = 1;
    //    for (int frameNumber = person.startFrameNumber; frameNumber <= person.endFrameNumber; frameNumber += stepSize)
    //    {
    //        std::vector<const TransformFrame*> samples;
    //        const TransformFrame* currentTransform = nullptr;
    //        for (const auto& hrf : person.headOrientationTrack)
    //        {
    //            if (hrf.frameNumber < frameNumber - windowSize ||
    //                hrf.frameNumber > frameNumber + windowSize)
    //            {
    //                continue;
    //            }
    //            else if (hrf.frameNumber == frameNumber)
    //            {
    //                currentTransform = &hrf;
    //            }
    //            samples.push_back(&hrf);
    //        }

    //        if (!currentTransform)
    //            continue;

    //        auto translationGood = false;
    //        auto rotationGood = false;
    //        cv::Point3f mean{};
    //        float variance{};

    //        // translation
    //        for (const auto& sample : samples)
    //        {
    //            mean += sample->translation;
    //        }
    //        mean /= float(samples.size());

    //        for (const auto& sample : samples)
    //        {
    //            variance += squareLengthPoint3(sample->translation - mean);
    //        }
    //        variance /= float(samples.size());

    //        auto deviation = squareLengthPoint3(currentTransform->translation - mean);
    //        if (deviation <= variance * deviationThreshold)
    //        {
    //            translationGood = true;
    //        }

    //        // rotation
    //        mean = cv::Point3f{};
    //        variance = 0.f;
    //        for (const auto& sample : samples)
    //        {
    //            mean += sample->rotation;
    //        }
    //        mean /= float(samples.size());

    //        for (const auto& sample : samples)
    //        {
    //            variance += squareLengthPoint3(sample->rotation - mean);
    //        }
    //        variance /= float(samples.size());

    //        deviation = squareLengthPoint3(currentTransform->rotation - mean);
    //        if (deviation <= variance * deviationThreshold)
    //        {
    //            rotationGood = true;
    //        }

    //        if (translationGood && rotationGood)
    //        {
    //            person.headOrientationTrack.push_back(*currentTransform);
    //        }
    //        else
    //        {
    //            auto droppedFrameNumber = frameNumber;
    //        }
    //    }
    //}

    // processing raw frames into key frames
    for (auto& p : persons)
    {
        auto& person = *p;

        auto nRawFrameTypes = person.frames.begin()->second.size();

        for (auto iRawFrameType = 0; iRawFrameType < nRawFrameTypes; ++iRawFrameType)
        {
            std::vector<std::vector<std::vector<KeyFrame>>> keypointTracks;
            const auto nKeypoints = person.frames.begin()->second[iRawFrameType].size();

            bool validTrack = false;
            for (auto iKeypointId = 0; iKeypointId < nKeypoints; ++iKeypointId)
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
                        else if (lastFrameNumber != frameNumber - 1)
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
                            for (auto lastFrameNumber = frameNumber - 1; lastFrameNumber > lastKeyframeNumber; --lastFrameNumber)
                            {
                                if (person.frames.find(lastFrameNumber) != person.frames.end())
                                {
                                    const auto& lastFrame = person.frames[lastFrameNumber][iRawFrameType][iKeypointId];
                                    if (lastFrame.confidence > 0.f)
                                    {
                                        if (nFrames == 0)
                                        {
                                            lastValidFrameNumber = lastFrameNumber;
                                        }

                                        cv::Point2f lastFramePos(lastFrame.x, lastFrame.y);

                                        float t = float(lastFrameNumber - lastKeyframeNumber) / float(frameNumber - lastKeyframeNumber);
                                        cv::Point2f estimatedPos = lastKeyFramePos * (1 - t) + currentFramePos * t;

                                        mse += squareLengthPoint2(estimatedPos - lastFramePos);
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
            else if (lastFrame->frameNumber != frameNumber - 1)
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
                int nFrames = 0;
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

                    mseTranslation += squareLengthPoint3(estimatedTranslation - tt);
                    mseRotation += squareLengthPoint3(estimatedRotation - rr);
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
    std::vector<nlohmann::json> jRoots;
    int iPerson = 0;
    while (iPerson < persons.size())
    {
        jRoots.push_back(nlohmann::json{});
        nlohmann::json jPersons;
        int numKeyFrames = 0;
        std::vector<PersonInfo*> personsThisSegment;
        int startFrameNumberSegment = std::numeric_limits<int>::max();
        int endFrameNumberSegment = 0;
        for ( ; iPerson < persons.size(); ++iPerson)
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
                    nlohmann::json jTracks;

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

                    jTracks["type"] = keypointName;

                    nlohmann::json jKeypoints;
                    for (int iKeypoint = 0; iKeypoint < kvt.second.size(); ++iKeypoint)
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
                        jKeypoints[iKeypoint] = jKeyFrameTracks;
                    }

                    jTracks["keypoints"] = jKeypoints;
                    jPerson["tracks"].push_back(jTracks);
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
            jPerson["faceTranslation"] = jFaceTranslationTracks;

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

            jPerson["faceRotation"] = jFaceRotationTracks;

            jPersons.push_back(jPerson);

            personsThisSegment.push_back(pp);

            numKeyFrames += p.numKeyFrames;

            if (numKeyFrames >= maxFramesPerSegment || iPerson >= persons.size() - 1)
            {
                jRoots.back()["persons"] = jPersons;

                //int numIntervals = int(ceil(float(endFrameNumberSegment - startFrameNumberSegment + 1) / indexerInterval));
                //nlohmann::json jIndexer;
                //for (auto i = 0; i < numIntervals; ++i)
                //{
                //    nlohmann::json jInterval;
                //    int startFrameNumber = i * indexerInterval + startFrameNumberSegment;
                //    int endFrameNumber = std::min(startFrameNumber + indexerInterval, int(mVideoNumFrames));

                //    nlohmann::json jPersonsInterval;
                //    for (int j = 0; j < personsThisSegment.size(); ++j)
                //    {
                //        auto& pp = personsThisSegment[j];
                //        auto& p = *pp;

                //        if ((p.startFrameNumber >= startFrameNumber && p.startFrameNumber <= endFrameNumber) ||
                //            (p.endFrameNumber >= startFrameNumber && p.endFrameNumber <= endFrameNumber) ||
                //            (p.startFrameNumber <= startFrameNumber && p.endFrameNumber >= endFrameNumber))
                //        {
                //            jPersonsInterval.push_back(j);
                //        }
                //    }
                //    jInterval["start"] = startFrameNumber;
                //    jInterval["end"] = endFrameNumber;
                //    jInterval["persons"] = jPersonsInterval;

                //    jIndexer.push_back(jInterval);
                //}
                //jRoots.back()["indexer"] = jIndexer;
                //jRoots.back()["indexerInterval"] = indexerInterval;

                ++iPerson;
                break;
            }
        }
    }

    for (int i = 0; i < jRoots.size(); ++i)
    {
        auto& jRoot = jRoots[i];
        jRoot["numSegments"] = jRoots.size();
        jRoot["segmentIndex"] = i;
        std::stringstream ss;
        ss << this->mJsonPath << "_" << i;
        std::ofstream ofile(ss.str() + ".json");
        ofile << jRoot;
        ofile.close();

        std::string command = "7z.exe a " + ss.str() + ".7z " + ss.str() + ".json";
        system(command.c_str());
    }
}

}