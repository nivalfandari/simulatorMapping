 //
// Created by tzuk on 6/4/23.
//

#include "simulator.h"

cv::Mat Simulator::getCurrentLocation() {
    locationLock.lock();
    cv::Mat locationCopy = Tcw.clone();
    locationLock.unlock();
    return locationCopy;
}

Simulator::Simulator(std::string ORBSLAMConfigFile, std::string model_path, bool alignModelToTexture,
                     std::string modelTextureNameToAlignTo, bool trackImages,
                     bool saveMap, std::string simulatorOutputDirPath, bool loadMap, std::string mapLoadPath,
                     double movementFactor,
                     double speedFactor,
                     std::string vocPath) : stopFlag(false), ready(false), saveMapSignal(false),
                                            track(false), start(false),
                                            movementFactor(movementFactor), modelPath(model_path),
                                            modelTextureNameToAlignTo(modelTextureNameToAlignTo),
                                            isSaveMap(saveMap),
                                            trackImages(trackImages), cull_backfaces(false), isInitalized(false),
                                            stopFlagSLAM(false),
                                            speedFactor(speedFactor),
                                            viewportDesiredSize(640, 480) {
    cv::FileStorage fSettings(ORBSLAMConfigFile, cv::FileStorage::READ);

    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    this->numberOfFeatures = fSettings["ORBextractor.nFeatures"];
    this->trackingNumberOfFeatures = fSettings["ORBextractor.trackingNumberOfFeatures"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    SLAM = std::make_shared<ORB_SLAM2::System>(vocPath, ORBSLAMConfigFile, ORB_SLAM2::System::MONOCULAR, true,
                                               loadMap, mapLoadPath, true);

    K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
    orbExtractor = std::make_shared<ORB_SLAM2::ORBextractor>(numberOfFeatures, fScaleFactor, nLevels, fIniThFAST,
                                                             fMinThFAST);
}

void Simulator::command(const std::string &command, int intervalUsleep, double fps, int totalCommandTimeInSeconds) {
    std::istringstream iss(command);
    std::string c;
    double value;
    iss >> c;
    if (commandMap.count(c) && commandMap[c]) {

        std::string stringValue;
        iss >> stringValue;
        value = std::stod(stringValue);
        applyCommand(c, value, intervalUsleep, fps, totalCommandTimeInSeconds);
    } else {
        std::cout << "the command " << c << " is not supported and will be skipped" << std::endl;
    }
}

bool Simulator::feedSLAM(cv::Mat &img) {
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto value = now_ms.time_since_epoch();
    double timestamp = value.count() / 1000.0;
    locationLock.lock();
    if (trackImages) {
        Tcw = SLAM->TrackMonocular(img, timestamp);
    } else {
        std::vector<cv::KeyPoint> pts;
        cv::Mat mDescriptors;
        orbExtractor->operator()(img, cv::Mat(), pts, mDescriptors);
        Tcw = SLAM->TrackMonocular(mDescriptors, pts, timestamp, img);
    }
    locationLock.unlock();
    auto state = SLAM->GetTracker()->mState;
    bool currentIsInitalized = !(state == SLAM->GetTracker()->NOT_INITIALIZED ||
                                 state == SLAM->GetTracker()->NO_IMAGES_YET);
    if (currentIsInitalized && !isInitalized) {
        SLAM->GetTracker()->SetNFeaturesToExtractor(trackingNumberOfFeatures);
    }
    if (!currentIsInitalized && isInitalized) {
        SLAM->GetTracker()->SetNFeaturesToExtractor(numberOfFeatures);
    }
    isInitalized = currentIsInitalized;
    return state == SLAM->GetTracker()->OK;
}

void Simulator::SLAMThread() {
    while (!stopFlagSLAM && !stopFlag) {
        if (track) {
            if (!currentImg.empty()) {
                imgLock.lock();
                cv::Mat img = currentImg.clone();
                imgLock.unlock();
                isLocalized = feedSLAM(img);
            } else {
                ORB_SLAM2::System::systemUsleep(1000);
            }
        }
    }

}

void Simulator::simulatorRunThread_two(const std::string& modelName, const std::string& displayName) {
    // Create the first window with a unique context name "Model"
    std::string windowName1 = "Model";
    pangolin::CreateWindowAndBind(windowName1);
    glEnable(GL_DEPTH_TEST);
    s_cam = pangolin::OpenGlRenderState (
        pangolin::ProjectionMatrix(viewportDesiredSize(0), viewportDesiredSize(1), K(0, 0), K(1, 1), K(0, 2),
                                   K(1, 2), 0.1, 20),
        pangolin::ModelViewLookAt(0.1, -0.1, 0.3, 0, 0, 0, 0.0, -1.0,
                                  pangolin::AxisY));
    pangolin::View &d_cam1 = pangolin::Display("simulator_d_cam1")
                             .SetBounds(0.0, 1.0, 0.0, 1.0, ((float)-viewportDesiredSize[0] / (float)viewportDesiredSize[1]))
                             .SetHandler(new pangolin::Handler3D(s_cam));

    bool show_bounds = false;
    bool show_axis = false;
    bool show_x0 = false;
    bool show_y0 = false;
    bool show_z0 = false;
    pangolin::RegisterKeyPressCallback('b', [&]() { show_bounds = !show_bounds; });
    pangolin::RegisterKeyPressCallback('0', [&]() { cull_backfaces = !cull_backfaces; });
    pangolin::RegisterKeyPressCallback('a', [&]() { show_axis = !show_axis; });
    pangolin::RegisterKeyPressCallback('k', [&]() { stopFlag = !stopFlag; });
    pangolin::RegisterKeyPressCallback('t', [&]() { track = !track; });
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_KEY_TAB, [&]() { start = true; });
    pangolin::RegisterKeyPressCallback('m', [&]() { saveMapSignal = !saveMapSignal; });
    pangolin::RegisterKeyPressCallback('x', [&]() { show_x0 = !show_x0; });
    pangolin::RegisterKeyPressCallback('y', [&]() { show_y0 = !show_y0; });
    pangolin::RegisterKeyPressCallback('z', [&]() { show_z0 = !show_z0; });
    pangolin::RegisterKeyPressCallback('w', [&]() { applyForwardToModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('a', [&]() { applyRightToModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('s', [&]() { applyForwardToModelCam(s_cam, -movementFactor); });
    pangolin::RegisterKeyPressCallback('d', [&]() { applyRightToModelCam(s_cam, -movementFactor); });
    pangolin::RegisterKeyPressCallback('e', [&]() { applyYawRotationToModelCam(s_cam, 1); });
    pangolin::RegisterKeyPressCallback('q', [&]() { applyYawRotationToModelCam(s_cam, -1); });
    pangolin::RegisterKeyPressCallback('r', [&]() { applyUpModelCam(s_cam, -movementFactor); }); // ORBSLAM y axis is reversed
    pangolin::RegisterKeyPressCallback('f', [&]() { applyUpModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('1', [&]() { slower(); });
    pangolin::RegisterKeyPressCallback('2', [&]() { faster(); });

    auto LoadProgram1 = [&]() {
        program.ClearShaders();
        program.AddShader(pangolin::GlSlAnnotatedShader, pangolin::shader);
        program.Link();
    };
    LoadProgram1();
    pangolin::ShowFullscreen(pangolin::TrueFalseToggle::True);
    pangolin::ShowFullscreen(pangolin::TrueFalseToggle::False);
    pangolin::VideoPixelFormat fmt1 = pangolin::VideoFormatFromString("RGBA32");
    int width1 = d_cam1.v.w;
    int height1 = d_cam1.v.h;

    pangolin::FinishFrame();


    // Create the second window with a unique context name "SecondWindow"
    std::string windowName2 = "SecondWindow";
    pangolin::CreateWindowAndBind(windowName2, 640, 480);
    glEnable(GL_DEPTH_TEST);
    pangolin::OpenGlRenderState s_cam2(
        pangolin::ProjectionMatrix(viewportDesiredSize(0), viewportDesiredSize(1), K(0, 0), K(1, 1), K(0, 2),
                                   K(1, 2), 0.1, 20),
        pangolin::ModelViewLookAt(0.1, -0.1, 0.3, 0, 0, 0, 0.0, -1.0,
                                  pangolin::AxisY));
    pangolin::View &d_cam2 = pangolin::Display("simulator_d_cam2")
                             .SetBounds(0.0, 1.0, 0.0, 1.0, ((float)-viewportDesiredSize[0] / (float)viewportDesiredSize[1]))
                             .SetHandler(new pangolin::Handler3D(s_cam2));
    pangolin::FinishFrame();

    int numberOfFramesForOrbslam = 0;
    const pangolin::Geometry modelGeometry = pangolin::LoadGeometry(modelPath);
    if (alignModelToTexture) {
        std::cout << "inside" << std::endl;
        alignModelViewPointToSurface(modelGeometry, modelTextureNameToAlignTo);
    }
    geomToRender = pangolin::ToGlGeometry(modelGeometry);
    for (auto& buffer : geomToRender.buffers) {
        buffer.second.attributes.erase("normal");
    }

    std::thread slamThread;
    slamThread = std::move(std::thread(&Simulator::SLAMThread, this));

    // Main loop
    while (!pangolin::ShouldQuit() && !stopFlag) {
        // Render content for the first window
        ready = true;
        pangolin::BindToContext(windowName1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (d_cam1.IsShown()) {
            d_cam1.Activate(s_cam);
            // Render your content for the first window here
            pangolin::glDrawColouredCube();
            /*if (cull_backfaces) {
                glEnable(GL_CULL_FACE);
                glCullFace(GL_BACK);
            }


            program.Bind();
            program.SetUniform("KT_cw", s_cam.GetProjectionMatrix() * s_cam.GetModelViewMatrix());
            pangolin::GlDraw(program, geomToRender, nullptr);
            program.Unbind();
            std::vector<unsigned char> buffer(4 * width1 * height1);
            glReadPixels(0, 0, width1, height1, GL_RGBA, GL_UNSIGNED_BYTE, buffer.data());
            cv::Mat imgBuffer = cv::Mat(height1, width1, CV_8UC4, buffer.data());
            imgLock.lock();
            cv::cvtColor(imgBuffer, currentImg, cv::COLOR_RGBA2GRAY);
            cv::flip(currentImg, currentImg, 0);
            imgLock.unlock();
            s_cam.Apply();


            glDisable(GL_CULL_FACE);
            pangolin::FinishFrame();


            if (saveMapSignal) {
                saveMapSignal = false;
                char time_buf[21];
                time_t now_t;
                std::time(&now_t);
                std::strftime(time_buf, 21, "%Y-%m-%d_%H:%S:%MZ", gmtime(&now_t));
                std::string currentTime(time_buf);
                saveMap(currentTime);
                SLAM->SaveMap(simulatorOutputDir + "/simulatorCloudPoint" + currentTime + ".bin");
                std::cout << "new map saved to " << simulatorOutputDir + "/simulatorCloudPoint" + currentTime + ".bin"
                    << std::endl;
            }

            //             drawPoints(seenPoints, keypoint_points);
       */ }
        /*else {
            s_cam.Apply();
            pangolin::FinishFrame();

        }*/
        
        pangolin::FinishFrame();

        
        // Render content for the second window
        pangolin::BindToContext(windowName2);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (d_cam2.IsShown()) {
            d_cam2.Activate(s_cam2);
            // Render your content for the second window here
            pangolin::glDrawColouredCube();
        }
        pangolin::FinishFrame();
    }
    if (isSaveMap) {

        saveMap("final");
        SLAM->SaveMap(simulatorOutputDir + "/finalSimulatorCloudPoint.bin");
        std::cout << "new map saved to " << simulatorOutputDir + "/finalSimulatorCloudPoint.bin" << std::endl;
    }

    // Clean up and shutdown
    SLAM->Shutdown();
}

void Simulator::simulatorRunThread(const std::string& modelName, const std::string& displayName) {
    std::string windowName1 = modelName;
    std::string windowName2 = "model2";
    pangolin::CreateWindowAndBind(windowName1);


    // we manually need to restore the properties of the context
    glEnable(GL_DEPTH_TEST);
    s_cam = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(viewportDesiredSize(0), viewportDesiredSize(1), K(0, 0), K(1, 1), K(0, 2),
            K(1, 2), 0.1, 20),
        pangolin::ModelViewLookAt(0.1, -0.1, 0.3, 0, 0, 0, 0.0, -1.0,
            pangolin::AxisY)); // the first 3 value are meaningless because we change them later


//pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam1 = pangolin::Display(displayName+"_left")
        .SetBounds(0.0, 1.0, 0.0, 1.0, ((float)-viewportDesiredSize[0] / (float)viewportDesiredSize[1]))
        .SetHandler(new pangolin::Handler3D(s_cam));
    //.SetHandler(&handler);
    pangolin::View& d_cam2 = pangolin::Display(displayName+"_right")
        .SetBounds(0.0, 1.0, 0.0, 1.0, ((float)-viewportDesiredSize[0] / (float)viewportDesiredSize[1]))
        .SetHandler(new pangolin::Handler3D(s_cam));

    bool show_bounds = false;
    bool show_axis = false;
    bool show_x0 = false;
    bool show_y0 = false;
    bool show_z0 = false;
    pangolin::RegisterKeyPressCallback('b', [&]() { show_bounds = !show_bounds; });
    pangolin::RegisterKeyPressCallback('0', [&]() { cull_backfaces = !cull_backfaces; });
    pangolin::RegisterKeyPressCallback('a', [&]() { show_axis = !show_axis; });
    pangolin::RegisterKeyPressCallback('k', [&]() { stopFlag = !stopFlag; });
    pangolin::RegisterKeyPressCallback('t', [&]() { track = !track; });
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_KEY_TAB, [&]() { start = true; });
    pangolin::RegisterKeyPressCallback('m', [&]() { saveMapSignal = !saveMapSignal; });
    pangolin::RegisterKeyPressCallback('x', [&]() { show_x0 = !show_x0; });
    pangolin::RegisterKeyPressCallback('y', [&]() { show_y0 = !show_y0; });
    pangolin::RegisterKeyPressCallback('z', [&]() { show_z0 = !show_z0; });
    pangolin::RegisterKeyPressCallback('w', [&]() { applyForwardToModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('a', [&]() { applyRightToModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('s', [&]() { applyForwardToModelCam(s_cam, -movementFactor); });
    pangolin::RegisterKeyPressCallback('d', [&]() { applyRightToModelCam(s_cam, -movementFactor); });
    pangolin::RegisterKeyPressCallback('e', [&]() { applyYawRotationToModelCam(s_cam, 1); });
    pangolin::RegisterKeyPressCallback('q', [&]() { applyYawRotationToModelCam(s_cam, -1); });
    pangolin::RegisterKeyPressCallback('r', [&]() { applyUpModelCam(s_cam, -movementFactor); }); // ORBSLAM y axis is reversed
    pangolin::RegisterKeyPressCallback('f', [&]() { applyUpModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('1', [&]() { slower(); });
    pangolin::RegisterKeyPressCallback('2', [&]() { faster(); });
    auto LoadProgram = [&]() {
        program.ClearShaders();
        program.AddShader(pangolin::GlSlAnnotatedShader, pangolin::shader);
        program.Link();
    };
    LoadProgram();


    pangolin::FinishFrame();

    int numberOfFramesForOrbslam = 0;
    const pangolin::Geometry modelGeometry = pangolin::LoadGeometry(modelPath);
    alignModelViewPointToSurface(modelGeometry, modelTextureNameToAlignTo);
    geomToRender = pangolin::ToGlGeometry(modelGeometry);
    for (auto& buffer : geomToRender.buffers) {
        buffer.second.attributes.erase("normal");
    }
    pangolin::ShowFullscreen(pangolin::TrueFalseToggle::True);
    pangolin::ShowFullscreen(pangolin::TrueFalseToggle::False);
    pangolin::VideoPixelFormat fmt = pangolin::VideoFormatFromString("RGBA32");
    int width1 = d_cam1.v.w;
    int height1 = d_cam1.v.h;
    int width2 = d_cam2.v.w;
    int height2 = d_cam2.v.h;
    std::thread slamThread;
    slamThread = std::move(std::thread(&Simulator::SLAMThread, this));
    while (!pangolin::ShouldQuit() && !stopFlag) {
        pangolin::BindToContext(modelName);
        ready = true;
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam1.Activate();
        if (d_cam1.IsShown()) {
            d_cam1.Activate(s_cam);
            if (cull_backfaces) {
                glEnable(GL_CULL_FACE);
                glCullFace(GL_BACK);
            }


            program.Bind();
            program.SetUniform("KT_cw", s_cam.GetProjectionMatrix() * s_cam.GetModelViewMatrix());
            pangolin::GlDraw(program, geomToRender, nullptr);
            program.Unbind();
            std::vector<unsigned char> buffer(4 * width1 * height1);
            glReadPixels(0, 0, width1, height1, GL_RGBA, GL_UNSIGNED_BYTE, buffer.data());
            cv::Mat imgBuffer = cv::Mat(height1, width1, CV_8UC4, buffer.data());
            imgLock.lock();
            cv::cvtColor(imgBuffer, currentImg, cv::COLOR_RGBA2GRAY);
            cv::flip(currentImg, currentImg, 0);
            imgLock.unlock();
            s_cam.Apply();


            glDisable(GL_CULL_FACE);
            pangolin::FinishFrame();


            if (saveMapSignal) {
                saveMapSignal = false;
                char time_buf[21];
                time_t now_t;
                std::time(&now_t);
                std::strftime(time_buf, 21, "%Y-%m-%d_%H:%S:%MZ", gmtime(&now_t));
                std::string currentTime(time_buf);
                saveMap(currentTime);
                SLAM->SaveMap(simulatorOutputDir + "/simulatorCloudPoint" + currentTime + ".bin");
                std::cout << "new map saved to " << simulatorOutputDir + "/simulatorCloudPoint" + currentTime + ".bin"
                    << std::endl;
            }

            //             drawPoints(seenPoints, keypoint_points);
        }
        else {
            s_cam.Apply();
            pangolin::FinishFrame();

        }

        d_cam2.Activate();
        if (d_cam2.IsShown()) {
            d_cam2.Activate(s_cam);
            if (cull_backfaces) {
                glEnable(GL_CULL_FACE);
                glCullFace(GL_BACK);
            }


            program.Bind();
            program.SetUniform("KT_cw", s_cam.GetProjectionMatrix() * s_cam.GetModelViewMatrix());
            pangolin::GlDraw(program, geomToRender, nullptr);
            program.Unbind();
            std::vector<unsigned char> buffer(4 * width2 * height2);
            glReadPixels(0, 0, width2, height2, GL_RGBA, GL_UNSIGNED_BYTE, buffer.data());
            cv::Mat imgBuffer = cv::Mat(height2, width2, CV_8UC4, buffer.data());
            imgLock.lock();
            cv::cvtColor(imgBuffer, currentImg, cv::COLOR_RGBA2GRAY);
            cv::flip(currentImg, currentImg, 0);
            imgLock.unlock();
            s_cam.Apply();


            glDisable(GL_CULL_FACE);
            pangolin::FinishFrame();


            if (saveMapSignal) {
                saveMapSignal = false;
                char time_buf[21];
                time_t now_t;
                std::time(&now_t);
                std::strftime(time_buf, 21, "%Y-%m-%d_%H:%S:%MZ", gmtime(&now_t));
                std::string currentTime(time_buf);
                saveMap(currentTime);
                SLAM->SaveMap(simulatorOutputDir + "/simulatorCloudPoint" + currentTime + ".bin");
                std::cout << "new map saved to " << simulatorOutputDir + "/simulatorCloudPoint" + currentTime + ".bin"
                    << std::endl;
            }

            //             drawPoints(seenPoints, keypoint_points);
        }
        else {
            s_cam.Apply();
            pangolin::FinishFrame();

        }

    }
    if (isSaveMap) {

        saveMap("final");
        SLAM->SaveMap(simulatorOutputDir + "/finalSimulatorCloudPoint.bin");
        std::cout << "new map saved to " << simulatorOutputDir + "/finalSimulatorCloudPoint.bin" << std::endl;
    }
    SLAM->Shutdown();
}

void Simulator::simulatorRunThread_orig(const std::string& modelName, const std::string& displayName) {
    std::string windowName1 = modelName;
    std::string windowName2 = "model2";
    pangolin::CreateWindowAndBind(windowName1);
    

    // we manually need to restore the properties of the context
    glEnable(GL_DEPTH_TEST);
    s_cam = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(viewportDesiredSize(0), viewportDesiredSize(1), K(0, 0), K(1, 1), K(0, 2),
                                       K(1, 2), 0.1, 20),
            pangolin::ModelViewLookAt(0.1, -0.1, 0.3, 0, 0, 0, 0.0, -1.0,
                                      pangolin::AxisY)); // the first 3 value are meaningless because we change them later

    //pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::Display(displayName)
        .SetBounds(0.0, 1.0, 0.0, 1.0, ((float)-viewportDesiredSize[0] / (float)viewportDesiredSize[1]))
        .SetHandler(new pangolin::Handler3D(s_cam));
    //.SetHandler(&handler);

    bool show_bounds = false;
    bool show_axis = false;
    bool show_x0 = false;
    bool show_y0 = false;
    bool show_z0 = false;
    pangolin::RegisterKeyPressCallback('b', [&]() { show_bounds = !show_bounds; });
    pangolin::RegisterKeyPressCallback('0', [&]() { cull_backfaces = !cull_backfaces; });
    pangolin::RegisterKeyPressCallback('a', [&]() { show_axis = !show_axis; });
    pangolin::RegisterKeyPressCallback('k', [&]() { stopFlag = !stopFlag; });
    pangolin::RegisterKeyPressCallback('t', [&]() { track = !track; });
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_KEY_TAB, [&]() { start = true; });
    pangolin::RegisterKeyPressCallback('m', [&]() { saveMapSignal = !saveMapSignal; });
    pangolin::RegisterKeyPressCallback('x', [&]() { show_x0 = !show_x0; });
    pangolin::RegisterKeyPressCallback('y', [&]() { show_y0 = !show_y0; });
    pangolin::RegisterKeyPressCallback('z', [&]() { show_z0 = !show_z0; });
    pangolin::RegisterKeyPressCallback('w', [&]() { applyForwardToModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('a', [&]() { applyRightToModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('s', [&]() { applyForwardToModelCam(s_cam, -movementFactor); });
    pangolin::RegisterKeyPressCallback('d', [&]() { applyRightToModelCam(s_cam, -movementFactor); });
    pangolin::RegisterKeyPressCallback('e', [&]() { applyYawRotationToModelCam(s_cam, 1); });
    pangolin::RegisterKeyPressCallback('q', [&]() { applyYawRotationToModelCam(s_cam, -1); });
    pangolin::RegisterKeyPressCallback('r', [&]() { applyUpModelCam(s_cam, -movementFactor); }); // ORBSLAM y axis is reversed
    pangolin::RegisterKeyPressCallback('f', [&]() { applyUpModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('1', [&]() { slower(); });
    pangolin::RegisterKeyPressCallback('2', [&]() { faster(); });
    auto LoadProgram = [&]() {
        program.ClearShaders();
        program.AddShader(pangolin::GlSlAnnotatedShader, pangolin::shader);
        program.Link();
    };
    LoadProgram();

            
    pangolin::FinishFrame();
            
    int numberOfFramesForOrbslam = 0;
    const pangolin::Geometry modelGeometry = pangolin::LoadGeometry(modelPath);
    if (alignModelToTexture) {
        alignModelViewPointToSurface(modelGeometry, modelTextureNameToAlignTo);
    }
    geomToRender = pangolin::ToGlGeometry(modelGeometry);
    for (auto &buffer: geomToRender.buffers) {
        buffer.second.attributes.erase("normal");
    }
    pangolin::ShowFullscreen(pangolin::TrueFalseToggle::True);
    pangolin::ShowFullscreen(pangolin::TrueFalseToggle::False);
    pangolin::VideoPixelFormat fmt = pangolin::VideoFormatFromString("RGBA32");
    int width = d_cam.v.w;
    int height = d_cam.v.h;
    std::thread slamThread;
    slamThread = std::move(std::thread(&Simulator::SLAMThread, this));
    while (!pangolin::ShouldQuit() && !stopFlag) {
    	pangolin::BindToContext(modelName);
        ready = true;
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (d_cam.IsShown()) {
            d_cam.Activate(s_cam);
            if (cull_backfaces) {
                glEnable(GL_CULL_FACE);
                glCullFace(GL_BACK);
            }


            program.Bind();
            program.SetUniform("KT_cw", s_cam.GetProjectionMatrix() * s_cam.GetModelViewMatrix());
            pangolin::GlDraw(program, geomToRender, nullptr);
            program.Unbind();
            std::vector<unsigned char> buffer(4 * width * height);
            glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer.data());
            cv::Mat imgBuffer = cv::Mat(height, width, CV_8UC4, buffer.data());
            imgLock.lock();
            cv::cvtColor(imgBuffer, currentImg, cv::COLOR_RGBA2GRAY);
            cv::flip(currentImg, currentImg, 0);
            imgLock.unlock();
            s_cam.Apply();


            glDisable(GL_CULL_FACE);
            pangolin::FinishFrame();


            if (saveMapSignal) {
                saveMapSignal = false;
                char time_buf[21];
                time_t now_t;
                std::time(&now_t);
                std::strftime(time_buf, 21, "%Y-%m-%d_%H:%S:%MZ", gmtime(&now_t));
                std::string currentTime(time_buf);
                saveMap(currentTime);
                SLAM->SaveMap(simulatorOutputDir + "/simulatorCloudPoint" + currentTime + ".bin");
                std::cout << "new map saved to " << simulatorOutputDir + "/simulatorCloudPoint" + currentTime + ".bin"
                          << std::endl;
            }

            //             drawPoints(seenPoints, keypoint_points);
        } else {
            s_cam.Apply();
            pangolin::FinishFrame();

        }

    }
    if (isSaveMap) {

        saveMap("final");
        SLAM->SaveMap(simulatorOutputDir + "/finalSimulatorCloudPoint.bin");
        std::cout << "new map saved to " << simulatorOutputDir + "/finalSimulatorCloudPoint.bin" << std::endl;
    }
    SLAM->Shutdown();
}

void Simulator::simulatorRunThread_orig2() {
    std::string windowName = "Model";
    pangolin::CreateWindowAndBind(windowName);

    // we manually need to restore the properties of the context
    glEnable(GL_DEPTH_TEST);
    s_cam = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(viewportDesiredSize(0), viewportDesiredSize(1), K(0, 0), K(1, 1), K(0, 2),
            K(1, 2), 0.1, 20),
        pangolin::ModelViewLookAt(0.1, -0.1, 0.3, 0, 0, 0, 0.0, -1.0,
            pangolin::AxisY)); // the first 3 value are meaningless because we change them later

    bool show_bounds = false;
    bool show_axis = false;
    bool show_x0 = false;
    bool show_y0 = false;
    bool show_z0 = false;
    pangolin::RegisterKeyPressCallback('b', [&]() { show_bounds = !show_bounds; });
    pangolin::RegisterKeyPressCallback('0', [&]() { cull_backfaces = !cull_backfaces; });
    pangolin::RegisterKeyPressCallback('a', [&]() { show_axis = !show_axis; });
    pangolin::RegisterKeyPressCallback('k', [&]() { stopFlag = !stopFlag; });
    pangolin::RegisterKeyPressCallback('t', [&]() { track = !track; });
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_KEY_TAB, [&]() { start = true; });
    pangolin::RegisterKeyPressCallback('m', [&]() { saveMapSignal = !saveMapSignal; });
    pangolin::RegisterKeyPressCallback('x', [&]() { show_x0 = !show_x0; });
    pangolin::RegisterKeyPressCallback('y', [&]() { show_y0 = !show_y0; });
    pangolin::RegisterKeyPressCallback('z', [&]() { show_z0 = !show_z0; });
    pangolin::RegisterKeyPressCallback('w', [&]() { applyForwardToModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('a', [&]() { applyRightToModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('s', [&]() { applyForwardToModelCam(s_cam, -movementFactor); });
    pangolin::RegisterKeyPressCallback('d', [&]() { applyRightToModelCam(s_cam, -movementFactor); });
    pangolin::RegisterKeyPressCallback('e', [&]() { applyYawRotationToModelCam(s_cam, 1); });
    pangolin::RegisterKeyPressCallback('q', [&]() { applyYawRotationToModelCam(s_cam, -1); });
    pangolin::RegisterKeyPressCallback('r', [&]() { applyUpModelCam(s_cam, -movementFactor); }); // ORBSLAM y axis is reversed
    pangolin::RegisterKeyPressCallback('f', [&]() { applyUpModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('1', [&]() { slower(); });
    pangolin::RegisterKeyPressCallback('2', [&]() { faster(); });
    auto LoadProgram = [&]() {
        program.ClearShaders();
        program.AddShader(pangolin::GlSlAnnotatedShader, pangolin::shader);
        program.Link();
    };
    LoadProgram();
    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::Display("simulator_d_cam")
        .SetBounds(0.0, 1.0, 0.0, 1.0, ((float)-viewportDesiredSize[0] / (float)viewportDesiredSize[1]))
        .SetHandler(&handler);
    int numberOfFramesForOrbslam = 0;
    const pangolin::Geometry modelGeometry = pangolin::LoadGeometry(modelPath);
    //if (alignModelToTexture) {
    //    alignModelViewPointToSurface(modelGeometry, modelTextureNameToAlignTo);
    //}
    geomToRender = pangolin::ToGlGeometry(modelGeometry);
    for (auto& buffer : geomToRender.buffers) {
        buffer.second.attributes.erase("normal");
    }
    pangolin::ShowFullscreen(pangolin::TrueFalseToggle::True);
    pangolin::ShowFullscreen(pangolin::TrueFalseToggle::False);
    pangolin::VideoPixelFormat fmt = pangolin::VideoFormatFromString("RGBA32");
    int width = d_cam.v.w;
    int height = d_cam.v.h;
    std::thread slamThread;
    slamThread = std::move(std::thread(&Simulator::SLAMThread, this));
    while (!pangolin::ShouldQuit() && !stopFlag) {
        ready = true;
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (d_cam.IsShown()) {
            d_cam.Activate(s_cam);
            if (cull_backfaces) {
                glEnable(GL_CULL_FACE);
                glCullFace(GL_BACK);
            }


            program.Bind();
            program.SetUniform("KT_cw", s_cam.GetProjectionMatrix() * s_cam.GetModelViewMatrix());
            pangolin::GlDraw(program, geomToRender, nullptr);
            program.Unbind();
            std::vector<unsigned char> buffer(4 * width * height);
            glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer.data());
            cv::Mat imgBuffer = cv::Mat(height, width, CV_8UC4, buffer.data());
            imgLock.lock();
            cv::cvtColor(imgBuffer, currentImg, cv::COLOR_RGBA2GRAY);
            cv::flip(currentImg, currentImg, 0);
            imgLock.unlock();
            s_cam.Apply();


            glDisable(GL_CULL_FACE);
            pangolin::FinishFrame();


            if (saveMapSignal) {
                saveMapSignal = false;
                char time_buf[21];
                time_t now_t;
                std::time(&now_t);
                std::strftime(time_buf, 21, "%Y-%m-%d_%H:%S:%MZ", gmtime(&now_t));
                std::string currentTime(time_buf);
                saveMap(currentTime);
                SLAM->SaveMap(simulatorOutputDir + "/simulatorCloudPoint" + currentTime + ".bin");
                std::cout << "new map saved to " << simulatorOutputDir + "/simulatorCloudPoint" + currentTime + ".bin"
                    << std::endl;
            }

            //             drawPoints(seenPoints, keypoint_points);
        }
        else {
            s_cam.Apply();
            pangolin::FinishFrame();

        }

    }
    if (isSaveMap) {

        saveMap("final");
        SLAM->SaveMap(simulatorOutputDir + "/finalSimulatorCloudPoint.bin");
        std::cout << "new map saved to " << simulatorOutputDir + "/finalSimulatorCloudPoint.bin" << std::endl;
    }
    SLAM->Shutdown();
}

void Simulator::simulatorRunThread_multi(const std::string& modelName, const std::string& displayName) {

    pangolin::CreateWindowAndBind("Main Window", 1280, 480);

    // Enable depth testing for 3D rendering
    glEnable(GL_DEPTH_TEST);

    // Create two views (displays) for parallel visualization
    pangolin::View& left_display = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 0.5)
        .SetAspect(640.0 / 480.0);

    pangolin::View& right_display = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.5, 1.0)
        .SetAspect(640.0 / 480.0);
    s_cam = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(viewportDesiredSize(0), viewportDesiredSize(1), K(0, 0), K(1, 1), K(0, 2),
            K(1, 2), 0.1, 20),
        pangolin::ModelViewLookAt(0.1, -0.1, 0.3, 0, 0, 0, 0.0, -1.0,
            pangolin::AxisY));

     pangolin::OpenGlRenderState s_cam2(
        pangolin::ProjectionMatrix(viewportDesiredSize(0), viewportDesiredSize(1), K(0, 0), K(1, 1), K(0, 2),
            K(1, 2), 0.1, 20),
        pangolin::ModelViewLookAt(1, 1, 1, 0, 0, 0, 0.0, 1.0,
            pangolin::AxisY));


     //pangolin::Handler3D handler(s_cam);
     pangolin::View& d_cam = pangolin::Display("simulator_d_cam")
         .SetBounds(0.0, 1.0, 0.0, 1.0, ((float)-viewportDesiredSize[0] / (float)viewportDesiredSize[1]))
         .SetHandler(new pangolin::Handler3D(s_cam));
/*
    pangolin::View& d_cam2 = pangolin::Display("simulator_d_cam2")
        .SetBounds(0.0, 1.0, 0.0, 1.0, ((float)-viewportDesiredSize[0] / (float)viewportDesiredSize[1]))
        .SetHandler(new pangolin::Handler3D(s_cam2));*/

     left_display.SetHandler(new pangolin::Handler3D(s_cam));
     right_display.SetHandler(new pangolin::Handler3D(s_cam2));

    bool show_bounds = false;
    bool show_axis = false;
    bool show_x0 = false;
    bool show_y0 = false;
    bool show_z0 = false;
    pangolin::RegisterKeyPressCallback('b', [&]() { show_bounds = !show_bounds; });
    pangolin::RegisterKeyPressCallback('0', [&]() { cull_backfaces = !cull_backfaces; });
    pangolin::RegisterKeyPressCallback('a', [&]() { show_axis = !show_axis; });
    pangolin::RegisterKeyPressCallback('k', [&]() { stopFlag = !stopFlag; });
    pangolin::RegisterKeyPressCallback('t', [&]() { track = !track; });
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_KEY_TAB, [&]() { start = true; });
    pangolin::RegisterKeyPressCallback('m', [&]() { saveMapSignal = !saveMapSignal; });
    pangolin::RegisterKeyPressCallback('x', [&]() { show_x0 = !show_x0; });
    pangolin::RegisterKeyPressCallback('y', [&]() { show_y0 = !show_y0; });
    pangolin::RegisterKeyPressCallback('z', [&]() { show_z0 = !show_z0; });
    pangolin::RegisterKeyPressCallback('w', [&]() { applyForwardToModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('a', [&]() { applyRightToModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('s', [&]() { applyForwardToModelCam(s_cam, -movementFactor); });
    pangolin::RegisterKeyPressCallback('d', [&]() { applyRightToModelCam(s_cam, -movementFactor); });
    pangolin::RegisterKeyPressCallback('e', [&]() { applyYawRotationToModelCam(s_cam, 1); });
    pangolin::RegisterKeyPressCallback('q', [&]() { applyYawRotationToModelCam(s_cam, -1); });
    pangolin::RegisterKeyPressCallback('r', [&]() { applyUpModelCam(s_cam, -movementFactor); }); // ORBSLAM y axis is reversed
    pangolin::RegisterKeyPressCallback('f', [&]() { applyUpModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('1', [&]() { slower(); });
    pangolin::RegisterKeyPressCallback('2', [&]() { faster(); });


    pangolin::RegisterKeyPressCallback('8', [&]() { applyForwardToModelCam(s_cam2, movementFactor); });
    pangolin::RegisterKeyPressCallback('u', [&]() { applyRightToModelCam(s_cam2, movementFactor); });
    pangolin::RegisterKeyPressCallback('i', [&]() { applyForwardToModelCam(s_cam2, -movementFactor); });
    pangolin::RegisterKeyPressCallback('o', [&]() { applyRightToModelCam(s_cam2, -movementFactor); });
    pangolin::RegisterKeyPressCallback('7', [&]() { applyYawRotationToModelCam(s_cam2, 1); });
    pangolin::RegisterKeyPressCallback('u', [&]() { applyYawRotationToModelCam(s_cam2, -1); });
    pangolin::RegisterKeyPressCallback('[', [&]() { applyUpModelCam(s_cam2, -movementFactor); }); // ORBSLAM y axis is reversed
    pangolin::RegisterKeyPressCallback(']', [&]() { applyUpModelCam(s_cam2, movementFactor); });

    bool together = true;
    pangolin::RegisterKeyPressCallback('g', [&]() { together = !together; });

    auto LoadProgram = [&]() {
        program.ClearShaders();
        program.AddShader(pangolin::GlSlAnnotatedShader, pangolin::shader);
        program.Link();
    };
    LoadProgram();



    int numberOfFramesForOrbslam = 0;
    const pangolin::Geometry modelGeometry = pangolin::LoadGeometry(modelPath);
    if (alignModelToTexture) {
        //std::cout << "inside" << std::endl;
        alignModelViewPointToSurface(modelGeometry, modelTextureNameToAlignTo);
    }
    geomToRender = pangolin::ToGlGeometry(modelGeometry);
    for (auto& buffer : geomToRender.buffers) {
        buffer.second.attributes.erase("normal");
    }
    pangolin::ShowFullscreen(pangolin::TrueFalseToggle::True);
    pangolin::ShowFullscreen(pangolin::TrueFalseToggle::False);
    pangolin::VideoPixelFormat fmt = pangolin::VideoFormatFromString("RGBA32");
    int width = d_cam.v.w;
    int height = d_cam.v.h;
    std::thread slamThread;
    slamThread = std::move(std::thread(&Simulator::SLAMThread, this));




    // Main rendering loop
    while (!pangolin::ShouldQuit() && !stopFlag) {
        ready = true;
        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (together) {
            // Update camera_right to match camera_left parameters
            s_cam2.SetModelViewMatrix(s_cam.GetModelViewMatrix());
            s_cam2.SetProjectionMatrix(s_cam.GetProjectionMatrix());
            //applyRightToModelCam(s_cam2, 1);
        }

        // Activate and render to the left display
        left_display.Activate(s_cam);
        // Your rendering code for the left display goes here
        //pangolin::glDrawColouredCube();  // Example rendering function
        if (cull_backfaces) {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
        }


        program.Bind();
        program.SetUniform("KT_cw", s_cam.GetProjectionMatrix() * s_cam.GetModelViewMatrix());
        pangolin::GlDraw(program, geomToRender, nullptr);
        program.Unbind();
        std::vector<unsigned char> buffer(4 * width * height);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer.data());
        cv::Mat imgBuffer = cv::Mat(height, width, CV_8UC4, buffer.data());
        imgLock.lock();
        cv::cvtColor(imgBuffer, currentImg, cv::COLOR_RGBA2GRAY);
        cv::flip(currentImg, currentImg, 0);
        imgLock.unlock();
        s_cam.Apply();


        glDisable(GL_CULL_FACE);
        //pangolin::FinishFrame();


        if (saveMapSignal) {
            saveMapSignal = false;
            char time_buf[21];
            time_t now_t;
            std::time(&now_t);
            std::strftime(time_buf, 21, "%Y-%m-%d_%H:%S:%MZ", gmtime(&now_t));
            std::string currentTime(time_buf);
            saveMap(currentTime);
            SLAM->SaveMap(simulatorOutputDir + "/simulatorCloudPoint" + currentTime + ".bin");
            std::cout << "new map saved to " << simulatorOutputDir + "/simulatorCloudPoint" + currentTime + ".bin"
                << std::endl;
        }

        // Activate and render to the right display
        right_display.Activate(s_cam2);
        // Your rendering code for the right display goes here
        //pangolin::glDrawColouredCube();  // Example rendering function

        if (cull_backfaces) {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
        }


        program.Bind();
        program.SetUniform("KT_cw", s_cam2.GetProjectionMatrix() * s_cam2.GetModelViewMatrix());
        pangolin::GlDraw(program, geomToRender, nullptr);
        program.Unbind();
        std::vector<unsigned char> buffer2(4 * width * height);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer2.data());
        cv::Mat imgBuffer2 = cv::Mat(height, width, CV_8UC4, buffer2.data());
        imgLock.lock();
        cv::cvtColor(imgBuffer2, currentImg, cv::COLOR_RGBA2GRAY);
        cv::flip(currentImg, currentImg, 0);
        imgLock.unlock();
        s_cam2.Apply();


        glDisable(GL_CULL_FACE);
        //pangolin::FinishFrame();


        if (saveMapSignal) {
            saveMapSignal = false;
            char time_buf[21];
            time_t now_t;
            std::time(&now_t);
            std::strftime(time_buf, 21, "%Y-%m-%d_%H:%S:%MZ", gmtime(&now_t));
            std::string currentTime(time_buf);
            saveMap(currentTime);
            SLAM->SaveMap(simulatorOutputDir + "/simulatorCloudPoint" + currentTime + ".bin");
            std::cout << "new map saved to " << simulatorOutputDir + "/simulatorCloudPoint" + currentTime + ".bin"
                << std::endl;
        }

        // Swap buffers and process events
        pangolin::FinishFrame();
    }
    if (isSaveMap) {

        saveMap("final");
        SLAM->SaveMap(simulatorOutputDir + "/finalSimulatorCloudPoint.bin");
        std::cout << "new map saved to " << simulatorOutputDir + "/finalSimulatorCloudPoint.bin" << std::endl;
    }
    SLAM->Shutdown();
}

void Simulator::simulatorRunThread_nerf(const std::string& modelName, const std::string& displayName) {

    pangolin::CreateWindowAndBind("Main Window", 1280, 480);

    // Enable depth testing for 3D rendering
    glEnable(GL_DEPTH_TEST);

    // Create two views (displays) for parallel visualization
    pangolin::View& left_display = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 0.5)
        .SetAspect(640.0 / 480.0);

    pangolin::View& right_display = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.5, 1.0)
        .SetAspect(640.0 / 480.0);
    s_cam = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(viewportDesiredSize(0), viewportDesiredSize(1), K(0, 0), K(1, 1), K(0, 2),
            K(1, 2), 0.1, 20),
        pangolin::ModelViewLookAt(0.1, -0.1, 0.3, 0, 0, 0, 0.0, -1.0,
            pangolin::AxisY));

    pangolin::OpenGlRenderState s_cam2(
        pangolin::ProjectionMatrix(viewportDesiredSize(0), viewportDesiredSize(1), K(0, 0), K(1, 1), K(0, 2),
            K(1, 2), 0.1, 20),
        pangolin::ModelViewLookAt(1, 1, 1, 0, 0, 0, 0.0, 1.0,
            pangolin::AxisY));


    //pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::Display("simulator_d_cam")
        .SetBounds(0.0, 1.0, 0.0, 1.0, ((float)-viewportDesiredSize[0] / (float)viewportDesiredSize[1]))
        .SetHandler(new pangolin::Handler3D(s_cam));
    /*
        pangolin::View& d_cam2 = pangolin::Display("simulator_d_cam2")
            .SetBounds(0.0, 1.0, 0.0, 1.0, ((float)-viewportDesiredSize[0] / (float)viewportDesiredSize[1]))
            .SetHandler(new pangolin::Handler3D(s_cam2));*/

    left_display.SetHandler(new pangolin::Handler3D(s_cam));
    right_display.SetHandler(new pangolin::Handler3D(s_cam2));

    bool show_bounds = false;
    bool show_axis = false;
    bool show_x0 = false;
    bool show_y0 = false;
    bool show_z0 = false;
    pangolin::RegisterKeyPressCallback('b', [&]() { show_bounds = !show_bounds; });
    pangolin::RegisterKeyPressCallback('0', [&]() { cull_backfaces = !cull_backfaces; });
    pangolin::RegisterKeyPressCallback('a', [&]() { show_axis = !show_axis; });
    pangolin::RegisterKeyPressCallback('k', [&]() { stopFlag = !stopFlag; });
    pangolin::RegisterKeyPressCallback('t', [&]() { track = !track; });
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_KEY_TAB, [&]() { start = true; });
    pangolin::RegisterKeyPressCallback('m', [&]() { saveMapSignal = !saveMapSignal; });
    pangolin::RegisterKeyPressCallback('x', [&]() { show_x0 = !show_x0; });
    pangolin::RegisterKeyPressCallback('y', [&]() { show_y0 = !show_y0; });
    pangolin::RegisterKeyPressCallback('z', [&]() { show_z0 = !show_z0; });
    pangolin::RegisterKeyPressCallback('w', [&]() { applyForwardToModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('a', [&]() { applyRightToModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('s', [&]() { applyForwardToModelCam(s_cam, -movementFactor); });
    pangolin::RegisterKeyPressCallback('d', [&]() { applyRightToModelCam(s_cam, -movementFactor); });
    pangolin::RegisterKeyPressCallback('e', [&]() { applyYawRotationToModelCam(s_cam, 1); });
    pangolin::RegisterKeyPressCallback('q', [&]() { applyYawRotationToModelCam(s_cam, -1); });
    pangolin::RegisterKeyPressCallback('r', [&]() { applyUpModelCam(s_cam, -movementFactor); }); // ORBSLAM y axis is reversed
    pangolin::RegisterKeyPressCallback('f', [&]() { applyUpModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('1', [&]() { slower(); });
    pangolin::RegisterKeyPressCallback('2', [&]() { faster(); });


    pangolin::RegisterKeyPressCallback('8', [&]() { applyForwardToModelCam(s_cam2, movementFactor); });
    pangolin::RegisterKeyPressCallback('u', [&]() { applyRightToModelCam(s_cam2, movementFactor); });
    pangolin::RegisterKeyPressCallback('i', [&]() { applyForwardToModelCam(s_cam2, -movementFactor); });
    pangolin::RegisterKeyPressCallback('o', [&]() { applyRightToModelCam(s_cam2, -movementFactor); });
    pangolin::RegisterKeyPressCallback('7', [&]() { applyYawRotationToModelCam(s_cam2, 1); });
    pangolin::RegisterKeyPressCallback('u', [&]() { applyYawRotationToModelCam(s_cam2, -1); });
    pangolin::RegisterKeyPressCallback('[', [&]() { applyUpModelCam(s_cam2, -movementFactor); }); // ORBSLAM y axis is reversed
    pangolin::RegisterKeyPressCallback(']', [&]() { applyUpModelCam(s_cam2, movementFactor); });

    bool together = true;
    pangolin::RegisterKeyPressCallback('g', [&]() { together = !together; });

    auto LoadProgram = [&]() {
        program.ClearShaders();
        program.AddShader(pangolin::GlSlAnnotatedShader, pangolin::shader);
        program.Link();
    };
    LoadProgram();



    int numberOfFramesForOrbslam = 0;
    const pangolin::Geometry modelGeometry = pangolin::LoadGeometry(modelPath);
    //if (alignModelToTexture) {
        //std::cout << "inside" << std::endl;
        //alignModelViewPointToSurface(modelGeometry, modelTextureNameToAlignTo);
    //}
    applyPitchRotationToModelCam(s_cam, -90);
    geomToRender = pangolin::ToGlGeometry(modelGeometry);
    for (auto& buffer : geomToRender.buffers) {
        buffer.second.attributes.erase("normal");
    }
    pangolin::ShowFullscreen(pangolin::TrueFalseToggle::True);
    pangolin::ShowFullscreen(pangolin::TrueFalseToggle::False);
    pangolin::VideoPixelFormat fmt = pangolin::VideoFormatFromString("RGBA32");
    int width = d_cam.v.w;
    int height = d_cam.v.h;
    std::thread slamThread;
    slamThread = std::move(std::thread(&Simulator::SLAMThread, this));




    // Main rendering loop
    while (!pangolin::ShouldQuit() && !stopFlag) {
        ready = true;
        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (together) {
            // Update camera_right to match camera_left parameters
            s_cam2.SetModelViewMatrix(s_cam.GetModelViewMatrix());
            s_cam2.SetProjectionMatrix(s_cam.GetProjectionMatrix());
            //applyRightToModelCam(s_cam2, 1);
        }

        // Activate and render to the left display
        left_display.Activate(s_cam);
        // Your rendering code for the left display goes here
        //pangolin::glDrawColouredCube();  // Example rendering function
        if (cull_backfaces) {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
        }


        program.Bind();
        program.SetUniform("KT_cw", s_cam.GetProjectionMatrix() * s_cam.GetModelViewMatrix());
        pangolin::GlDraw(program, geomToRender, nullptr);
        program.Unbind();
        std::vector<unsigned char> buffer(4 * width * height);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer.data());
        cv::Mat imgBuffer = cv::Mat(height, width, CV_8UC4, buffer.data());
        imgLock.lock();
        cv::cvtColor(imgBuffer, currentImg, cv::COLOR_RGBA2GRAY);
        cv::flip(currentImg, currentImg, 0);
        imgLock.unlock();
        s_cam.Apply();


        glDisable(GL_CULL_FACE);
        //pangolin::FinishFrame();


        if (saveMapSignal) {
            saveMapSignal = false;
            char time_buf[21];
            time_t now_t;
            std::time(&now_t);
            std::strftime(time_buf, 21, "%Y-%m-%d_%H:%S:%MZ", gmtime(&now_t));
            std::string currentTime(time_buf);
            saveMap(currentTime);
            SLAM->SaveMap(simulatorOutputDir + "/simulatorCloudPoint" + currentTime + ".bin");
            std::cout << "new map saved to " << simulatorOutputDir + "/simulatorCloudPoint" + currentTime + ".bin"
                << std::endl;
        }

        // Activate and render to the right display
        right_display.Activate(s_cam2);
        // Your rendering code for the right display goes here
        //pangolin::glDrawColouredCube();  // Example rendering function

        if (cull_backfaces) {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
        }


        program.Bind();
        program.SetUniform("KT_cw", s_cam2.GetProjectionMatrix() * s_cam2.GetModelViewMatrix());
        pangolin::GlDraw(program, geomToRender, nullptr);
        program.Unbind();
        std::vector<unsigned char> buffer2(4 * width * height);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer2.data());
        cv::Mat imgBuffer2 = cv::Mat(height, width, CV_8UC4, buffer2.data());
        imgLock.lock();
        cv::cvtColor(imgBuffer2, currentImg, cv::COLOR_RGBA2GRAY);
        cv::flip(currentImg, currentImg, 0);
        imgLock.unlock();
        s_cam2.Apply();


        glDisable(GL_CULL_FACE);
        //pangolin::FinishFrame();


        if (saveMapSignal) {
            saveMapSignal = false;
            char time_buf[21];
            time_t now_t;
            std::time(&now_t);
            std::strftime(time_buf, 21, "%Y-%m-%d_%H:%S:%MZ", gmtime(&now_t));
            std::string currentTime(time_buf);
            saveMap(currentTime);
            SLAM->SaveMap(simulatorOutputDir + "/simulatorCloudPoint" + currentTime + ".bin");
            std::cout << "new map saved to " << simulatorOutputDir + "/simulatorCloudPoint" + currentTime + ".bin"
                << std::endl;
        }

        // Swap buffers and process events
        pangolin::FinishFrame();
    }
    if (isSaveMap) {

        saveMap("final");
        SLAM->SaveMap(simulatorOutputDir + "/finalSimulatorCloudPoint.bin");
        std::cout << "new map saved to " << simulatorOutputDir + "/finalSimulatorCloudPoint.bin" << std::endl;
    }
    SLAM->Shutdown();
}

std::thread Simulator::run(const std::string& modelName, const std::string& displayName) {
    std::thread thread(&Simulator::simulatorRunThread_orig, this, modelName, displayName);
    return thread;
}

void Simulator::saveMap(std::string prefix) {
    std::ofstream pointData;

    pointData.open(simulatorOutputDir + "/cloud" + prefix + ".csv");
    for (auto &p: SLAM->GetMap()->GetAllMapPoints()) {
        if (p != nullptr && !p->isBad()) {
            auto point = p->GetWorldPos();
            Eigen::Matrix<double, 3, 1> vector = ORB_SLAM2::Converter::toVector3d(point);
            cv::Mat worldPos = cv::Mat::zeros(3, 1, CV_64F);
            worldPos.at<double>(0) = vector.x();
            worldPos.at<double>(1) = vector.y();
            worldPos.at<double>(2) = vector.z();
            p->UpdateNormalAndDepth();
            cv::Mat Pn = p->GetNormal();
            Pn.convertTo(Pn, CV_64F);
            pointData << worldPos.at<double>(0) << "," << worldPos.at<double>(1) << "," << worldPos.at<double>(2);
            pointData << "," << p->GetMinDistanceInvariance() << "," << p->GetMaxDistanceInvariance() << ","
                      << Pn.at<double>(0) << "," << Pn.at<double>(1) << "," << Pn.at<double>(2);
            std::map<ORB_SLAM2::KeyFrame *, size_t> observations = p->GetObservations();
            for (auto &obs: observations) {
                ORB_SLAM2::KeyFrame *currentFrame = obs.first;
                if (!currentFrame->image.empty()) {
                    size_t pointIndex = obs.second;
                    cv::KeyPoint keyPoint = currentFrame->mvKeysUn[pointIndex];
                    cv::Point2f featurePoint = keyPoint.pt;
                    pointData << "," << currentFrame->mnId << "," << featurePoint.x << "," << featurePoint.y;
                }
            }
            pointData << std::endl;
        }
    }
    pointData.close();
}

void Simulator::extractSurface(const pangolin::Geometry &modelGeometry, std::string modelTextureNameToAlignTo,
                               Eigen::MatrixXf &surface) {
    std::vector<Eigen::Vector3<unsigned int>> surfaceIndices;
    for (auto &o: modelGeometry.objects) {
        if (o.first == modelTextureNameToAlignTo) {
            const auto &it_vert = o.second.attributes.find("vertex_indices");
            if (it_vert != o.second.attributes.end()) {
                const auto &vs = std::get<pangolin::Image<unsigned int>>(it_vert->second);
                for (size_t i = 0; i < vs.h; ++i) {
                    const Eigen::Map<const Eigen::Vector3<unsigned int>> v(vs.RowPtr(i));
                    surfaceIndices.emplace_back(v);
                }
            }
        }
    }
    surface = Eigen::MatrixXf(surfaceIndices.size() * 3, 3);
    int currentIndex = 0;
    for (const auto &b: modelGeometry.buffers) {
        const auto &it_vert = b.second.attributes.find("vertex");
        if (it_vert != b.second.attributes.end()) {
            const auto &vs = std::get<pangolin::Image<float>>(it_vert->second);
            for (auto &row: surfaceIndices) {
                for (auto &i: row) {
                    const Eigen::Map<const Eigen::Vector3f> v(vs.RowPtr(i));
                    surface.row(currentIndex++) = v;
                }
            }
        }
    }
}

void Simulator::applyPitchRotationToModelCam(pangolin::OpenGlRenderState &cam, double value) {
    double rand = double(value) * (M_PI / 180);
    double c = std::cos(rand);
    double s = std::sin(rand);

    Eigen::Matrix3d R;
    R << 1, 0, 0,
            0, c, -s,
            0, s, c;

    Eigen::Matrix4d pangolinR = Eigen::Matrix4d::Identity();;
    pangolinR.block<3, 3>(0, 0) = R;

    auto camMatrix = pangolin::ToEigen<double>(cam.GetModelViewMatrix());

    // Left-multiply the rotation
    camMatrix = pangolinR * camMatrix;

    // Convert back to pangolin matrix and set
    pangolin::OpenGlMatrix newModelView;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            newModelView.m[j * 4 + i] = camMatrix(i, j);
        }
    }

    cam.SetModelViewMatrix(newModelView);
}

void Simulator::intervalOverCommand(
        const std::function<void(pangolin::OpenGlRenderState &, double &)> &func, double value,
        int intervalUsleep, double fps, int totalCommandTimeInSeconds) {
    double intervalValue = this->speedFactor * value / (fps * totalCommandTimeInSeconds);
    int intervalIndex = 0;
    while (intervalIndex <= fps * totalCommandTimeInSeconds) {
        ORB_SLAM2::System::systemUsleep(intervalUsleep);
        func(s_cam, intervalValue);
        intervalIndex += 1;
    }
}

void Simulator::applyCommand(std::string &command, double value, int intervalUsleep, double fps,
                             int totalCommandTimeInSeconds) {
    if (command == "cw") {
        intervalOverCommand(Simulator::applyYawRotationToModelCam, value, intervalUsleep, fps,
                            totalCommandTimeInSeconds);
    } else if (command == "ccw") {
        intervalOverCommand(Simulator::applyYawRotationToModelCam, -1 * value,
                            intervalUsleep, fps,
                            totalCommandTimeInSeconds);
    } else if (command == "forward") {
        intervalOverCommand(Simulator::applyForwardToModelCam, value, intervalUsleep, fps,
                            totalCommandTimeInSeconds);
    } else if (command == "back") {
        intervalOverCommand(Simulator::applyForwardToModelCam, -1 * value, intervalUsleep,
                            fps, totalCommandTimeInSeconds);
    } else if (command == "right") {
        intervalOverCommand(Simulator::applyRightToModelCam, -1 * value, intervalUsleep,
                            fps, totalCommandTimeInSeconds);
    } else if (command == "left") {
        intervalOverCommand(Simulator::applyRightToModelCam, value, intervalUsleep, fps,
                            totalCommandTimeInSeconds);
    } else if (command == "up") {
        intervalOverCommand(Simulator::applyUpModelCam, -1 * value, intervalUsleep, fps,
                            totalCommandTimeInSeconds);
    } else if (command == "down") {
        intervalOverCommand(Simulator::applyUpModelCam, value, intervalUsleep, fps,
                            totalCommandTimeInSeconds);
    }
}

void Simulator::applyUpModelCam(pangolin::OpenGlRenderState &cam, double value) {
    auto camMatrix = pangolin::ToEigen<double>(cam.GetModelViewMatrix());
    camMatrix(1, 3) += value;
    cam.SetModelViewMatrix(camMatrix);
}

void Simulator::applyForwardToModelCam(pangolin::OpenGlRenderState &cam, double value) {
    auto camMatrix = pangolin::ToEigen<double>(cam.GetModelViewMatrix());
    camMatrix(2, 3) += value;
    cam.SetModelViewMatrix(camMatrix);
}

void Simulator::applyRightToModelCam(pangolin::OpenGlRenderState &cam, double value) {
    auto camMatrix = pangolin::ToEigen<double>(cam.GetModelViewMatrix());
    camMatrix(0, 3) += value;
    cam.SetModelViewMatrix(camMatrix);
}

void Simulator::applyYawRotationToModelCam(pangolin::OpenGlRenderState &cam, double value) {
    double rand = double(value) * (M_PI / 180);
    double c = std::cos(rand);
    double s = std::sin(rand);

    Eigen::Matrix3d R;
    R << c, 0, s,
            0, 1, 0,
            -s, 0, c;

    Eigen::Matrix4d pangolinR = Eigen::Matrix4d::Identity();
    pangolinR.block<3, 3>(0, 0) = R;

    auto camMatrix = pangolin::ToEigen<double>(cam.GetModelViewMatrix());

    // Left-multiply the rotation
    camMatrix = pangolinR * camMatrix;

    // Convert back to pangolin matrix and set
    pangolin::OpenGlMatrix newModelView;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            newModelView.m[j * 4 + i] = camMatrix(i, j);
        }
    }

    cam.SetModelViewMatrix(newModelView);
}

void Simulator::alignModelViewPointToSurface(const pangolin::Geometry &modelGeometry,
                                             std::string modelTextureNameToAlignTo) {
    Eigen::MatrixXf surface;
    extractSurface(modelGeometry, modelTextureNameToAlignTo, surface);
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(surface, Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.computeV();
    Eigen::Vector3f v = svd.matrixV().col(2);
    const auto mvm = pangolin::ModelViewLookAt(v.x(), v.y(), v.z(), 0, 0, 0, 0.0,
                                               -1.0,
                                               pangolin::AxisY);
    const auto proj = pangolin::ProjectionMatrix(viewportDesiredSize(0), viewportDesiredSize(1), K(0, 0), K(1, 1),
                                                 K(0, 2), K(1, 2), 0.1, 20);
    s_cam.SetModelViewMatrix(mvm);
    s_cam.SetProjectionMatrix(proj);
    applyPitchRotationToModelCam(s_cam, -90);
    //applyPitchRotationToModelCam(s_cam2, -90);
}

void Simulator::setSpeed(double speed)
{
    this->speedFactor = speed;
}

double Simulator::getSpeed() const
{
    return this->speedFactor;
}

void Simulator::faster()
{
    if(this->speedFactor < 3.0){ 
        this->speedFactor += 0.1;
    }
}

void Simulator::slower()
{
    if(this->speedFactor > 0.5){
        this->speedFactor -= 0.1;
    }
}
