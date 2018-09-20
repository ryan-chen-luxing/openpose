#include <openpose/wrapper/wrapperStructOutput.hpp>
#include <openpose/producer/videoReader.hpp>

namespace op
{
    WrapperStructOutput::WrapperStructOutput(const DisplayMode displayMode_, const bool guiVerbose_,
                                             const bool fullScreen_, const std::string& writeKeypoint_,
                                             const DataFormat writeKeypointFormat_, const std::shared_ptr<VideoReader> videoReader_,
                                             const std::string& writeJson_, std::size_t maxFramesPerSegment_,
                                             const std::string& writeCocoJson_, const std::string& writeCocoFootJson_,
                                             const std::string& writeImages_, const std::string& writeImagesFormat_,
                                             const std::string& writeVideo_, const double writeVideoFps_,
                                             const std::string& writeHeatMaps_,
                                             const std::string& writeHeatMapsFormat_,
                                             const std::string& writeVideoAdam_, const std::string& writeBvh_,
                                             const std::string& udpHost_, const std::string& udpPort_) :
        displayMode{displayMode_},
        guiVerbose{guiVerbose_},
        fullScreen{fullScreen_},
        writeKeypoint{writeKeypoint_},
        writeKeypointFormat{writeKeypointFormat_},
        videoReader{videoReader_},
        writeJson{writeJson_},
        maxFramesPerSegment{maxFramesPerSegment_},
        writeCocoJson{writeCocoJson_},
        writeCocoFootJson{writeCocoFootJson_},
        writeImages{writeImages_},
        writeImagesFormat{writeImagesFormat_},
        writeVideo{writeVideo_},
        writeHeatMaps{writeHeatMaps_},
        writeHeatMapsFormat{writeHeatMapsFormat_},
        writeVideoFps{writeVideoFps_},
        writeVideoAdam{writeVideoAdam_},
        writeBvh{writeBvh_},
        udpHost{udpHost_},
        udpPort{udpPort_}
    {
    }
}
