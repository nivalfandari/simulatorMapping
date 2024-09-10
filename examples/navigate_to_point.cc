#include <pangolin/utils/file_utils.h>
#include <pangolin/geometry/glgeometry.h>

#include "include/run_model/TextureShader.h"
#include "include/Auxiliary.h"

#define NEAR_PLANE 0.1
#define FAR_PLANE 20

#define ANIMATION_DURATION 5.0
#define ANIMATION_STEPS 100.0

void drawPoints(std::vector<cv::Point3d> seen_points, std::vector<cv::Point3d> new_points_seen) {
    std::string settingPath = Auxiliary::GetGeneralSettingsPath();
    std::ifstream programData(settingPath);
    nlohmann::json data;
    programData >> data;
    programData.close();

    const int point_size = data["pointSize"];

    glPointSize(point_size);
    glBegin(GL_POINTS);
    glColor3f(0.0, 0.0, 0.0);

    for (auto point : seen_points) {
        glVertex3f((float)(point.x), (float)(point.y), (float)(point.z));
    }
    glEnd();

    glPointSize(point_size);
    glBegin(GL_POINTS);
    glColor3f(1.0, 0.0, 0.0);

    for (auto point : new_points_seen) {
        glVertex3f((float)(point.x), (float)(point.y), (float)(point.z));
    }
    std::cout << new_points_seen.size() << std::endl;

    glEnd();
}

Eigen::Matrix4f loadMatrixFromFile(const std::string& filename) {
    Eigen::Matrix4f matrix;
    std::ifstream infile(filename);

    if (infile.is_open()) {
        int row = 0;
        std::string line;
        while (std::getline(infile, line)) {
            std::istringstream ss(line);
            std::string value;
            int col = 0;
            while (std::getline(ss, value, ',')) {
                matrix(row, col) = std::stof(value);
                col++;
            }
            row++;
        }
        infile.close();
    }
    else {
        std::cerr << "Cannot open file: " << filename << std::endl;
    }

    return matrix;
}

cv::Point3d transformPoint(const cv::Point3d& point, const Eigen::Matrix4f& transformation) {
    Eigen::Vector4f eigenPoint = Eigen::Vector4f((float)point.x, (float)point.y, (float)point.z, 1.0f);
    Eigen::Vector4f transformedPoint = transformation * eigenPoint;
    return cv::Point3d((double)transformedPoint(0), (double)transformedPoint(1), (double)transformedPoint(2));
}

std::vector<cv::Point3d> loadPoints() {
    std::ifstream pointData;
    std::vector<std::string> row;
    std::string line, word, temp;

    std::string settingPath = Auxiliary::GetGeneralSettingsPath();
    std::ifstream programData(settingPath);
    nlohmann::json data;
    programData >> data;
    programData.close();

    std::string cloud_points = std::string(data["mapInputDir"]) + "cloud1.csv";

    std::vector<cv::Point3d> points;

    pointData.open(cloud_points, std::ios::in);

    while (!pointData.eof()) {
        row.clear();

        std::getline(pointData, line);

        std::stringstream words(line);

        if (line == "") {
            continue;
        }

        while (std::getline(words, word, ',')) {
            try
            {
                std::stod(word);
            }
            catch (std::out_of_range)
            {
                word = "0";
            }
            row.push_back(word);
        }
        points.push_back(cv::Point3d(std::stod(row[0]), std::stod(row[1]), std::stod(row[2])));
    }
    pointData.close();

    return points;
}

void applyYawRotationToModelCam1(pangolin::OpenGlRenderState* s_cam, double value) {
    double rand = double(value) * (M_PI / 180);
    double c = std::cos(rand);
    double s = std::sin(rand);

    Eigen::Matrix3d R;
    R << c, 0, s,
        0, 1, 0,
        -s, 0, c;

    Eigen::Matrix4d pangolinR = Eigen::Matrix4d::Identity();
    pangolinR.block<3, 3>(0, 0) = R;

    auto camMatrix = pangolin::ToEigen<double>(s_cam->GetModelViewMatrix());

    // Left-multiply the rotation
    camMatrix = pangolinR * camMatrix;

    // Convert back to pangolin matrix and set
    pangolin::OpenGlMatrix newModelView;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            newModelView.m[j * 4 + i] = camMatrix(i, j);
        }
    }

    s_cam->SetModelViewMatrix(newModelView);
}

void applyRightToModelCam(pangolin::OpenGlRenderState& cam, double value) {
    auto camMatrix = pangolin::ToEigen<double>(cam.GetModelViewMatrix());
    camMatrix(0, 3) += value;
    cam.SetModelViewMatrix(camMatrix);
}

void applyUpModelCam(pangolin::OpenGlRenderState& cam, double value) {
    auto camMatrix = pangolin::ToEigen<double>(cam.GetModelViewMatrix());
    camMatrix(1, 3) += value;
    cam.SetModelViewMatrix(camMatrix);
}

void applyForwardToModelCam(pangolin::OpenGlRenderState& cam, double value) {
    auto camMatrix = pangolin::ToEigen<double>(cam.GetModelViewMatrix());
    camMatrix(2, 3) += value;
    cam.SetModelViewMatrix(camMatrix);
}

void applyYawRotationToModelCam(pangolin::OpenGlRenderState& cam, double value) {
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

void applyPitchRotationToModelCam(pangolin::OpenGlRenderState& cam, double value) {
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

void applyRollRotationToModelCam(pangolin::OpenGlRenderState& cam, double value) {
    double rand = double(value) * (M_PI / 180);
    double c = std::cos(rand);
    double s = std::sin(rand);

    Eigen::Matrix3d R;
    R << c, -s, 0,
        s, c, 0,
        0, 0, 1;

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

void rotate(pangolin::OpenGlRenderState& s_cam, double value, pangolin::Handler3D handler, pangolin::View& left_display, pangolin::View& right_display,
    pangolin::OpenGlRenderState& s_cam2, Eigen::Vector3d Pick_w, std::vector<Eigen::Vector3d> Picks_w, bool together, bool cull_backfaces, pangolin::GlSlProgram& default_prog,
    const pangolin::GlGeometry& geomToRender, bool is_nerf) {
    Eigen::Vector3d current_point;
    double step = value / ANIMATION_STEPS;
    for (int i = 1; i <= ANIMATION_STEPS; ++i) {

        applyYawRotationToModelCam(s_cam, step);

        // Render the scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if ((handler.Selected_P_w() - Pick_w).norm() > 1E-6) {
            Pick_w = handler.Selected_P_w();
            Picks_w.push_back(Pick_w);
            std::cout << pangolin::FormatString("\"Translation\": [%,%,%]", Pick_w[0], Pick_w[1], Pick_w[2])
                << std::endl;
        }

        if (together) {
            // Update camera_right to match camera_left parameters
            s_cam2.SetModelViewMatrix(s_cam.GetModelViewMatrix());
            s_cam2.SetProjectionMatrix(s_cam.GetProjectionMatrix());
            if (is_nerf)
                applyRightToModelCam(s_cam2, -0.2);
            else
                applyRightToModelCam(s_cam2, -0.5);
        }

        // Activate and render to the left display
        left_display.Activate(s_cam);


        if (cull_backfaces) {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
        }
        default_prog.Bind();
        default_prog.SetUniform("KT_cw", s_cam.GetProjectionMatrix() * s_cam.GetModelViewMatrix());
        pangolin::GlDraw(default_prog, geomToRender, nullptr);
        default_prog.Unbind();

        s_cam.Apply();

        glDisable(GL_CULL_FACE);


        // Activate and render to the right display
        right_display.Activate(s_cam2);

        if (cull_backfaces) {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
        }
        default_prog.Bind();
        default_prog.SetUniform("KT_cw", s_cam2.GetProjectionMatrix() * s_cam2.GetModelViewMatrix());
        pangolin::GlDraw(default_prog, geomToRender, nullptr);
        default_prog.Unbind();

        s_cam2.Apply();

        glDisable(GL_CULL_FACE);


        pangolin::FinishFrame();


        // Delay to control the animation speed
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

int main(int argc, char** argv) {
    std::string settingPath = Auxiliary::GetGeneralSettingsPath();
    std::ifstream programData(settingPath);
    nlohmann::json data;
    programData >> data;
    programData.close();

    std::string configPath = data["DroneYamlPathSlam"];
    cv::FileStorage fSettings(configPath, cv::FileStorage::READ);

    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    float viewpointX = fSettings["RunModel.ViewpointX"];
    float viewpointY = fSettings["RunModel.ViewpointY"];
    float viewpointZ = fSettings["RunModel.ViewpointZ"];

    Eigen::Matrix3d K;
    K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
    cv::Mat K_cv = (cv::Mat_<float>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    Eigen::Vector2i viewport_desired_size(640, 480);

    cv::Mat img;

    // Options
    bool show_bounds = false;
    bool show_axis = false;
    bool show_x0 = false;
    bool show_y0 = false;
    bool show_z0 = false;
    bool cull_backfaces = false;

    // Create Window for rendering
    pangolin::CreateWindowAndBind("Main", viewport_desired_size[0], viewport_desired_size[1]);
    glEnable(GL_DEPTH_TEST);

    // Create two views (displays) for parallel visualization
    pangolin::View& left_display = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 0.5)
        .SetAspect(640.0 / 480.0);

    pangolin::View& right_display = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.5, 1.0)
        .SetAspect(640.0 / 480.0);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(viewport_desired_size(0), viewport_desired_size(1), K(0, 0), K(1, 1), K(0, 2), K(1, 2), NEAR_PLANE, FAR_PLANE),
        pangolin::ModelViewLookAt(viewpointX, viewpointY, viewpointZ, 0, 0, 0, 0.0, -1.0, pangolin::AxisY)
    );

    pangolin::OpenGlRenderState s_cam2(
        pangolin::ProjectionMatrix(viewport_desired_size(0), viewport_desired_size(1), K(0, 0), K(1, 1), K(0, 2), K(1, 2), NEAR_PLANE, FAR_PLANE),
        pangolin::ModelViewLookAt(viewpointX, viewpointY, viewpointZ, 0, 0, 0, 0.0, -1.0, pangolin::AxisY)
    );

    // Create Interactive View in window
    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, ((float)-viewport_desired_size[0] / (float)viewport_desired_size[1]))
        .SetHandler(&handler);

    left_display.SetHandler(new pangolin::Handler3D(s_cam));
    right_display.SetHandler(new pangolin::Handler3D(s_cam2));

    // Load Geometry asynchronously
    std::string model_path = data["modelPath"];
    const pangolin::Geometry geom_to_load = pangolin::LoadGeometry(model_path);
    auto aabb = pangolin::GetAxisAlignedBox(geom_to_load);
    Eigen::AlignedBox3f total_aabb;
    total_aabb.extend(aabb);
    const auto mvm = pangolin::ModelViewLookAt(viewpointX, viewpointY, viewpointZ, 0, 0, 0, 0.0, -1.0, pangolin::AxisY);
    const auto proj = pangolin::ProjectionMatrix(viewport_desired_size(0), viewport_desired_size(1), K(0, 0), K(1, 1), K(0, 2), K(1, 2), NEAR_PLANE, FAR_PLANE);
    s_cam.SetModelViewMatrix(mvm);
    s_cam.SetProjectionMatrix(proj);
    s_cam2.SetModelViewMatrix(mvm);
    s_cam2.SetProjectionMatrix(proj);
    const pangolin::GlGeometry geomToRender = pangolin::ToGlGeometry(geom_to_load);

    applyUpModelCam(s_cam, -1);
    applyRightToModelCam(s_cam, 0.5);


    // Render tree for holding object position
    pangolin::GlSlProgram default_prog;
    auto LoadProgram = [&]() {
        default_prog.ClearShaders();
        default_prog.AddShader(pangolin::GlSlAnnotatedShader, pangolin::shader);
        default_prog.Link();
    };
    LoadProgram();
    pangolin::RegisterKeyPressCallback('b', [&]() { show_bounds = !show_bounds; });
    pangolin::RegisterKeyPressCallback('0', [&]() { cull_backfaces = !cull_backfaces; });

    // Show axis and axis planes
    pangolin::RegisterKeyPressCallback('a', [&]() { show_axis = !show_axis; });
    pangolin::RegisterKeyPressCallback('x', [&]() { show_x0 = !show_x0; });
    pangolin::RegisterKeyPressCallback('y', [&]() { show_y0 = !show_y0; });
    pangolin::RegisterKeyPressCallback('z', [&]() { show_z0 = !show_z0; });

    bool together = true;
    pangolin::RegisterKeyPressCallback('g', [&]() { together = !together; });

    cv::Mat Twc;
    bool use_lab_icp = bool(data["useLabICP"]);
    std::cout << use_lab_icp << std::endl;

    std::string transformation_matrix_csv_path;
    if (use_lab_icp)
    {
        transformation_matrix_csv_path = std::string(data["framesOutput"]) + "frames_lab_transformation_matrix.csv";
    }
    else
    {
        transformation_matrix_csv_path = std::string(data["framesOutput"]) + "frames_transformation_matrix.csv";
    }
    Eigen::Matrix4f transformation = loadMatrixFromFile(transformation_matrix_csv_path);
    std::cout << transformation << std::endl;


    double targetPointX = data["targetPointX"];
    double targetPointY = data["targetPointY"];
    double targetPointZ = data["targetPointZ"];


    std::vector<cv::Point3d> points_to_draw;
    std::vector<cv::Point3d> points = loadPoints();

    for (auto point : points) {
        points_to_draw.push_back(transformPoint(point, transformation));
    }

    const Eigen::Vector3d start_point(
        s_cam.GetModelViewMatrix().m[12],
        s_cam.GetModelViewMatrix().m[13],
        s_cam.GetModelViewMatrix().m[14]
    );

    Eigen::Vector3d Pick_w = handler.Selected_P_w();
    std::vector<Eigen::Vector3d> Picks_w;


    const float yaw_movement = data["yawMovement"];

    bool is_nerf = bool(data["nerf"]);


    double halftPointZ_board = (start_point[2] + 2.4) / 2;
    double halftPointZ_wall = (start_point[2] - 5) / 2;
    double temp;

    if (is_nerf) {

        pangolin::OpenGlMatrix& start_mat = s_cam.GetModelViewMatrix();
        start_mat.m[12] = -0.5;
        start_mat.m[13] = 0.7;
        start_mat.m[14] = 0;
        s_cam.SetModelViewMatrix(start_mat);
        applyYawRotationToModelCam(s_cam, 180);
        applyPitchRotationToModelCam(s_cam, -90);

        if (targetPointZ < 0)
        {
            applyPitchRotationToModelCam(s_cam, 10);
            rotate(s_cam, 180.0, handler, left_display, right_display, s_cam2, Pick_w, Picks_w, together, cull_backfaces, default_prog, geomToRender, is_nerf);
            targetPointX = -1 * targetPointX;
            targetPointZ = -1 * targetPointZ;
        }
        else {
            applyPitchRotationToModelCam(s_cam, 20);
            
        }
        
    }
    else {
        if (targetPointZ < halftPointZ_wall) {
            applyPitchRotationToModelCam(s_cam, -20);
            rotate(s_cam, 180.0, handler, left_display, right_display, s_cam2, Pick_w, Picks_w, together, cull_backfaces, default_prog, geomToRender, is_nerf);
            targetPointX = -1 * targetPointX;
            targetPointZ = -1 * targetPointZ + start_point[2];
            targetPointZ = targetPointZ - s_cam.GetModelViewMatrix().m[14];

        }
        else if (targetPointZ< halftPointZ_board && targetPointZ >halftPointZ_wall) {
            if (targetPointX > start_point[0]) {
                applyPitchRotationToModelCam(s_cam, -13);
                rotate(s_cam, -90.0, handler, left_display, right_display, s_cam2, Pick_w, Picks_w, together, cull_backfaces, default_prog, geomToRender, is_nerf);
                temp = targetPointZ;
                targetPointZ = targetPointX;
                targetPointX = -1 * temp;

            }
            else if (targetPointX < start_point[0]) {
                applyPitchRotationToModelCam(s_cam, -13);
                rotate(s_cam, 90.0, handler, left_display, right_display, s_cam2, Pick_w, Picks_w, together, cull_backfaces, default_prog, geomToRender, is_nerf);
                temp = targetPointZ;
                targetPointZ = -1 * targetPointX;
                targetPointX = temp;

            }
        }
    }

    const Eigen::Vector3d new_start_point(
        s_cam.GetModelViewMatrix().m[12],
        s_cam.GetModelViewMatrix().m[13],
        s_cam.GetModelViewMatrix().m[14]
    );

    std::cout << "start point: " << std::endl;
    std::cout << "X: " << start_point[0] << std::endl;
    std::cout << "Y: " << start_point[1] << std::endl;
    std::cout << "Z: " << start_point[2] << std::endl;
    std::cout << "new start point: " << std::endl;
    std::cout << "X: " << new_start_point[0] << std::endl;
    std::cout << "Y: " << new_start_point[1] << std::endl;
    std::cout << "Z: " << new_start_point[2] << std::endl;
    cv::Point3d target_point_transform(targetPointX, targetPointY, targetPointZ);
    std::cout << "target point: " << std::endl;
    std::cout << "X: " << targetPointX << std::endl;
    std::cout << "Y: " << targetPointY << std::endl;
    std::cout << "Z: " << targetPointZ << std::endl;
    target_point_transform = transformPoint(target_point_transform, transformation);
    std::cout << "target point transform: " << std::endl;
    std::cout << "X: " << target_point_transform.x << std::endl;
    std::cout << "Y: " << target_point_transform.y << std::endl;
    std::cout << "Z: " << target_point_transform.z << std::endl;

    const Eigen::Vector3d target_point(targetPointX, targetPointY, targetPointZ);
    Eigen::Vector3d step;
    if (is_nerf)
        step = (target_point - new_start_point) / ANIMATION_STEPS;
    else
        step = (target_point - start_point) / ANIMATION_STEPS;
    Eigen::Vector3d current_point;
    for (int i = 1; i <= ANIMATION_STEPS; ++i) {
        current_point[0] = new_start_point[0];
        current_point[1] = new_start_point[1] + step[1] * i;
        current_point[2] = new_start_point[2];

        // Update the camera's model view matrix
        pangolin::OpenGlMatrix& mv_matrix = s_cam.GetModelViewMatrix();
        mv_matrix.m[12] = current_point[0];
        mv_matrix.m[13] = current_point[1];
        mv_matrix.m[14] = current_point[2];

        // Render the scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if ((handler.Selected_P_w() - Pick_w).norm() > 1E-6) {
            Pick_w = handler.Selected_P_w();
            Picks_w.push_back(Pick_w);
            std::cout << pangolin::FormatString("\"Translation\": [%,%,%]", Pick_w[0], Pick_w[1], Pick_w[2])
                << std::endl;
        }

        if (together) {
            // Update camera_right to match camera_left parameters
            s_cam2.SetModelViewMatrix(s_cam.GetModelViewMatrix());
            s_cam2.SetProjectionMatrix(s_cam.GetProjectionMatrix());
            if (is_nerf)
                applyRightToModelCam(s_cam2, -0.2);
            else
                applyRightToModelCam(s_cam2, -0.5);
        }

        // Activate and render to the left display
        left_display.Activate(s_cam);

        if (cull_backfaces) {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
        }
        default_prog.Bind();
        default_prog.SetUniform("KT_cw", s_cam.GetProjectionMatrix() * s_cam.GetModelViewMatrix());
        pangolin::GlDraw(default_prog, geomToRender, nullptr);
        default_prog.Unbind();

        s_cam.Apply();

        glDisable(GL_CULL_FACE);

        // Activate and render to the right display
        right_display.Activate(s_cam2);

        if (cull_backfaces) {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
        }
        default_prog.Bind();
        default_prog.SetUniform("KT_cw", s_cam2.GetProjectionMatrix() * s_cam2.GetModelViewMatrix());
        pangolin::GlDraw(default_prog, geomToRender, nullptr);
        default_prog.Unbind();

        s_cam2.Apply();

        glDisable(GL_CULL_FACE);

        pangolin::FinishFrame();


        // Delay to control the animation speed
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    for (int i = 1; i <= ANIMATION_STEPS; ++i) {
        //Eigen::Vector3d current_point;
        current_point[0] = new_start_point[0] + step[0] * i;

        pangolin::OpenGlMatrix& mv_matrix = s_cam.GetModelViewMatrix();
        mv_matrix.m[12] = current_point[0];
        mv_matrix.m[13] = current_point[1];
        mv_matrix.m[14] = current_point[2];

        // Render the scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if ((handler.Selected_P_w() - Pick_w).norm() > 1E-6) {
            Pick_w = handler.Selected_P_w();
            Picks_w.push_back(Pick_w);
            std::cout << pangolin::FormatString("\"Translation\": [%,%,%]", Pick_w[0], Pick_w[1], Pick_w[2])
                << std::endl;
        }

        if (together) {
            // Update camera_right to match camera_left parameters
            s_cam2.SetModelViewMatrix(s_cam.GetModelViewMatrix());
            s_cam2.SetProjectionMatrix(s_cam.GetProjectionMatrix());
            if (is_nerf)
                applyRightToModelCam(s_cam2, -0.2);
            else
                applyRightToModelCam(s_cam2, -0.5);
        }

        // Activate and render to the left display
        left_display.Activate(s_cam);


        if (cull_backfaces) {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
        }
        default_prog.Bind();
        default_prog.SetUniform("KT_cw", s_cam.GetProjectionMatrix() * s_cam.GetModelViewMatrix());
        pangolin::GlDraw(default_prog, geomToRender, nullptr);
        default_prog.Unbind();

        s_cam.Apply();

        glDisable(GL_CULL_FACE);



        // Activate and render to the right display
        right_display.Activate(s_cam2);

        if (cull_backfaces) {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
        }
        default_prog.Bind();
        default_prog.SetUniform("KT_cw", s_cam2.GetProjectionMatrix() * s_cam2.GetModelViewMatrix());
        pangolin::GlDraw(default_prog, geomToRender, nullptr);
        default_prog.Unbind();

        s_cam2.Apply();

        glDisable(GL_CULL_FACE);

        pangolin::FinishFrame();


        // Delay to control the animation speed
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    for (int i = 1; i <= ANIMATION_STEPS; ++i) {
        current_point[2] = new_start_point[2] + step[2] * i;

        // Update the camera's model view matrix
        pangolin::OpenGlMatrix& mv_matrix = s_cam.GetModelViewMatrix();
        mv_matrix.m[12] = current_point[0];
        mv_matrix.m[13] = current_point[1];
        mv_matrix.m[14] = current_point[2];

        // Render the scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if ((handler.Selected_P_w() - Pick_w).norm() > 1E-6) {
            Pick_w = handler.Selected_P_w();
            Picks_w.push_back(Pick_w);
            std::cout << pangolin::FormatString("\"Translation\": [%,%,%]", Pick_w[0], Pick_w[1], Pick_w[2])
                << std::endl;
        }

        if (together) {
            // Update camera_right to match camera_left parameters
            s_cam2.SetModelViewMatrix(s_cam.GetModelViewMatrix());
            s_cam2.SetProjectionMatrix(s_cam.GetProjectionMatrix());
            if (is_nerf)
                applyRightToModelCam(s_cam2, -0.2);
            else
                applyRightToModelCam(s_cam2, -0.5);
        }

        // Activate and render to the left display
        left_display.Activate(s_cam);

        if (cull_backfaces) {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
        }
        default_prog.Bind();
        default_prog.SetUniform("KT_cw", s_cam.GetProjectionMatrix() * s_cam.GetModelViewMatrix());
        pangolin::GlDraw(default_prog, geomToRender, nullptr);
        default_prog.Unbind();

        s_cam.Apply();

        glDisable(GL_CULL_FACE);

        // Activate and render to the right display
        right_display.Activate(s_cam2);

        if (cull_backfaces) {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
        }
        default_prog.Bind();
        default_prog.SetUniform("KT_cw", s_cam2.GetProjectionMatrix() * s_cam2.GetModelViewMatrix());
        pangolin::GlDraw(default_prog, geomToRender, nullptr);
        default_prog.Unbind();

        s_cam2.Apply();

        glDisable(GL_CULL_FACE);

        pangolin::FinishFrame();


        // Delay to control the animation speed
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    while (!pangolin::ShouldQuit()) {
        if ((handler.Selected_P_w() - Pick_w).norm() > 1E-6) {
            Pick_w = handler.Selected_P_w();
            Picks_w.push_back(Pick_w);
            std::cout << pangolin::FormatString("\"Translation\": [%,%,%]", Pick_w[0], Pick_w[1], Pick_w[2])
                << std::endl;
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (together) {
            // Update camera_right to match camera_left parameters
            s_cam2.SetModelViewMatrix(s_cam.GetModelViewMatrix());
            s_cam2.SetProjectionMatrix(s_cam.GetProjectionMatrix());
            if (is_nerf)
                applyRightToModelCam(s_cam2, -0.2);
            else
                applyRightToModelCam(s_cam2, -0.5);
        }

        // Activate and render to the left display
        left_display.Activate(s_cam);

        // Load any pending geometry to the GPU.
        if (cull_backfaces) {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
        }
        default_prog.Bind();
        default_prog.SetUniform("KT_cw", s_cam.GetProjectionMatrix() * s_cam.GetModelViewMatrix());
        pangolin::GlDraw(default_prog, geomToRender, nullptr);
        default_prog.Unbind();

        s_cam.Apply();

        glDisable(GL_CULL_FACE);


        // Activate and render to the right display
        right_display.Activate(s_cam2);

        if (cull_backfaces) {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
        }
        default_prog.Bind();
        default_prog.SetUniform("KT_cw", s_cam2.GetProjectionMatrix() * s_cam2.GetModelViewMatrix());
        pangolin::GlDraw(default_prog, geomToRender, nullptr);
        default_prog.Unbind();

        s_cam2.Apply();

        glDisable(GL_CULL_FACE);

        pangolin::FinishFrame();
    }

    return 0;
}
