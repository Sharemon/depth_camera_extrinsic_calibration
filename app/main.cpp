/**
 * Copyright @2023 Sharemon. All rights reserved.
 *
 @author: sharemon
 @date: 2023-11-10
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

struct DepthIntrinsic
{
    int width;
    int height;
    double f;
    double cx;
    double cy;
    double camera_height;
    double pitch;
    double incline;
    double depth_max;
    double depth_noise;

    friend ostream &operator<<(ostream &out, DepthIntrinsic &intrinsic);
};

ostream &operator<<(ostream &out, DepthIntrinsic &intrinsic)
{
    out << "width: " << intrinsic.width << std::endl;
    out << "height: " << intrinsic.height << std::endl;
    out << "f: " << intrinsic.f << std::endl;
    out << "cx: " << intrinsic.cx << std::endl;
    out << "cy: " << intrinsic.cy << std::endl;
    out << "camera_height: " << intrinsic.camera_height << std::endl;
    out << "pitch: " << intrinsic.pitch << std::endl;
    out << "incline: " << intrinsic.incline << std::endl;
    out << "depth_max: " << intrinsic.depth_max << std::endl;
    out << "depth_noise: " << intrinsic.depth_noise << std::endl;

    return out;
}

void generate_ground_depth_map(string intrinsic_filepath, string depth_map_filepath, DepthIntrinsic &intrinsic)
{
    // read intrinsics
    FileStorage param_file(intrinsic_filepath, FileStorage::READ);

    param_file["width"] >> intrinsic.width;
    param_file["height"] >> intrinsic.height;
    param_file["f"] >> intrinsic.f;
    param_file["cx"] >> intrinsic.cx;
    param_file["cy"] >> intrinsic.cy;
    param_file["camera_height"] >> intrinsic.camera_height;
    param_file["pitch"] >> intrinsic.pitch;
    param_file["incline"] >> intrinsic.incline;
    param_file["depth_max"] >> intrinsic.depth_max;
    param_file["depth_noise"] >> intrinsic.depth_noise;

    cout << intrinsic << std::endl;

    // generate ground depth for each pixel
    double camera_height = intrinsic.camera_height;
    double pitch = -intrinsic.pitch * CV_PI / 180;
    double incline = intrinsic.incline * CV_PI / 180;
    // convert pitch-incline to rotation matrix
    Mat Rpitch = (Mat_<double>(3, 3) << 1, 0, 0, 0, cos(pitch), -sin(pitch), 0, sin(pitch), cos(pitch));
    Mat Rincline = (Mat_<double>(3, 3) << cos(incline), 0, sin(incline), 0, 1, 0, -sin(incline), 0, cos(incline));
    Mat R = Rpitch * Rincline; // intrinsic rotation
    // compute new optical axis
    Mat optical_axis = Mat::zeros(3, 1, CV_64FC1);
    optical_axis.at<double>(2, 0) = 1;
    Mat optical_axis_new = R * optical_axis;

    Mat depth_map = Mat::zeros(Size(intrinsic.width, intrinsic.height), CV_32FC1);
    for (int y = 0; y < intrinsic.height; y++)
    {
        for (int x = 0; x < intrinsic.width; x++)
        {
            Mat vector_from_pixel_to_center = (Mat_<double>(3, 1) << (x - intrinsic.cx) / intrinsic.f, (y - intrinsic.cy) / intrinsic.f, 0);
            Mat pixel_vector = optical_axis + vector_from_pixel_to_center;
            Mat pixel_vector_after_rotation = R * pixel_vector;
            pixel_vector_after_rotation /= norm(pixel_vector_after_rotation);

            double a = pixel_vector_after_rotation.at<double>(0, 0);
            double b = pixel_vector_after_rotation.at<double>(1, 0);
            double c = pixel_vector_after_rotation.at<double>(2, 0);

            if (b < DBL_EPSILON)
            {
                depth_map.at<float>(y, x) = 0;
            }
            else
            {
                double Y = camera_height;
                double X = Y / b * a;
                double Z = Y / b * c;

                Mat XYZ = (Mat_<double>(3, 1) << X, Y, Z);
                double d = XYZ.dot(optical_axis_new);

                depth_map.at<float>(y, x) = d < intrinsic.depth_max ? d : 0;
            }
        }
    }

    // save depth map
    FILE *fp = fopen(depth_map_filepath.c_str(), "wb");
    for (int y = 0; y < intrinsic.height; y++)
    {
        for (int x = 0; x < intrinsic.width; x++)
        {
            float d = depth_map.at<float>(y, x);

            fwrite(&d, sizeof(float), 1, fp);
        }
    }
    fclose(fp);
}

void fit_plane(Mat& depth, DepthIntrinsic& intrinsic)
{
    vector<Vec3d> points_3d;
	Vec3d pt_mean = { 0,0,0 };

    for (int y = 0; y < intrinsic.height; y++)
    {
        for (int x = 0; x < intrinsic.width; x++)
        {
            // read depth map
            double d = depth.at<float>(y, x);

            if (d > DBL_EPSILON)
            {
                Vec3d pt;
                pt[0] = (x - intrinsic.cx) / intrinsic.f * d;
                pt[1] = (y - intrinsic.cy) / intrinsic.f * d;
                pt[2] = d;

                points_3d.emplace_back(pt);
                pt_mean += pt;
            }
        }
    }

    pt_mean[0] /= points_3d.size();
    pt_mean[1] /= points_3d.size();
    pt_mean[2] /= points_3d.size();

    for (int i = 0; i < points_3d.size(); i++)
    {
        points_3d[i] -= pt_mean;
    }

    Mat pts_mat = Mat::zeros(points_3d.size(), 3, CV_64FC1);
    for (int i = 0; i < points_3d.size(); i++)
    {
        pts_mat.at<double>(i, 0) = points_3d[i][0];
        pts_mat.at<double>(i, 1) = points_3d[i][1];
        pts_mat.at<double>(i, 2) = points_3d[i][2];
    }

    Mat U, W, Vt;
    SVD::compute(pts_mat, W, U, Vt);

	double a = Vt.at<double>(Vt.rows - 1, 0);
    double b = Vt.at<double>(Vt.rows - 1, 1);
    double c = Vt.at<double>(Vt.rows - 1, 2);
    double d = a * pt_mean[0] + b * pt_mean[1] + c * pt_mean[2];
    
    std::cout << "plane parameters: " << std::endl;
    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
    std::cout << "c: " << c << std::endl;
    std::cout << "d: " << d << std::endl;

    // compute extrinsic
    // 可以看作用pitch-incline的旋转将[0,1,0]的地面法向量转换成了[a,b,c]
    intrinsic.camera_height = abs(d);

    double sign = d / abs(d);
    a *= sign;
    b *= sign;
    c *= sign;

    intrinsic.pitch = acos(b) / CV_PI * 180;
    intrinsic.incline = -atan(a/c) / CV_PI * 180;
}

void test_auto_extrinsic_calibration(string intrinsic_filepath, string depth_map_filepath, DepthIntrinsic &intrinsic)
{
    // read intrinsics
    FileStorage param_file(intrinsic_filepath, FileStorage::READ);

    param_file["width"] >> intrinsic.width;
    param_file["height"] >> intrinsic.height;
    param_file["f"] >> intrinsic.f;
    param_file["cx"] >> intrinsic.cx;
    param_file["cy"] >> intrinsic.cy;

    // read depth map
    Mat depth_map = Mat::zeros(Size(intrinsic.width, intrinsic.height), CV_32FC1);
    FILE *fp = fopen(depth_map_filepath.c_str(), "rb");

    for (int y = 0; y < intrinsic.height; y++)
    {
        for (int x = 0; x < intrinsic.width; x++)
        {
            // read depth map
            fread(&depth_map.at<float>(y,x), sizeof(float), 1, fp);
        }
    }
    fclose(fp);

    // fit plane
    fit_plane(depth_map, intrinsic);

    cout << "auto calibrate:" << std::endl;
    cout << "height: " << intrinsic.camera_height << std::endl;
    cout << "pitch: " << intrinsic.pitch << std::endl;
    cout << "incline: " << intrinsic.incline << std::endl;
}

void rotate_and_convert_depth_map_to_point_cloud(DepthIntrinsic &intrinsic, string depth_map_filepath, string point_cloud_filepath)
{
    // generate ground depth for each pixel
    double camera_height = intrinsic.camera_height;
    double pitch = -intrinsic.pitch * CV_PI / 180;
    double incline = intrinsic.incline * CV_PI / 180;
    // convert back
    // convert pitch-incline to rotation matrix
    Mat Rpitch = (Mat_<double>(3, 3) << 1, 0, 0, 0, cos(pitch), -sin(pitch), 0, sin(pitch), cos(pitch));
    Mat Rincline = (Mat_<double>(3, 3) << cos(incline), 0, sin(incline), 0, 1, 0, -sin(incline), 0, cos(incline));
    Mat R = Rpitch * Rincline; // intrinsic rotation

    ofstream point_cloud_file(point_cloud_filepath);
    FILE *fp = fopen(depth_map_filepath.c_str(), "rb");

    for (int y = 0; y < intrinsic.height; y++)
    {
        for (int x = 0; x < intrinsic.width; x++)
        {
            // read depth map
            float d = 0;
            fread(&d, sizeof(float), 1, fp);

            // convert depth map to point cloud and save
            if (d > DBL_EPSILON)
            {
                double X = (x - intrinsic.cx) / intrinsic.f * d;
                double Y = (y - intrinsic.cy) / intrinsic.f * d;
                double Z = d;

                Mat XYZ = (Mat_<double>(3, 1) << X, Y, Z);
                XYZ = R * XYZ;

                point_cloud_file << XYZ.at<double>(0, 0) << "," << XYZ.at<double>(1, 0) << "," << XYZ.at<double>(2, 0) << std::endl;
            }
        }
    }
    fclose(fp);
    point_cloud_file.close();
}

void convert_depth_map_to_point_cloud(DepthIntrinsic &intrinsic, string depth_map_filepath, string point_cloud_filepath)
{
    ofstream point_cloud_file(point_cloud_filepath);
    FILE *fp = fopen(depth_map_filepath.c_str(), "rb");

    for (int y = 0; y < intrinsic.height; y++)
    {
        for (int x = 0; x < intrinsic.width; x++)
        {
            // read depth map
            float d = 0;
            fread(&d, sizeof(float), 1, fp);

            // convert depth map to point cloud and save
            if (d != 0)
            {
                float X = (x - intrinsic.cx) / intrinsic.f * d;
                float Y = (y - intrinsic.cy) / intrinsic.f * d;
                float Z = d;

                point_cloud_file << X << "," << Y << "," << Z << std::endl;
            }
        }
    }
    fclose(fp);
    point_cloud_file.close();
}

int main(int argc, const char **argv)
{
    string intrinsic_filepath = "../data/depth_camera_params.yaml";
    string depth_map_filepath = "../data/depth_map.raw";
    string point_cloud_filepath = "../data/point_cloud.txt";

    DepthIntrinsic intrinsic;
    generate_ground_depth_map(intrinsic_filepath, depth_map_filepath, intrinsic);
    test_auto_extrinsic_calibration(intrinsic_filepath, depth_map_filepath, intrinsic);
    // test_smart_extrinsic_calibration();
    // smart 标定原理与auto标定原理一致，只不过用IMU的重力方向表示了地平面的法向量
    // smart 标定的产线标定中使用的IMU校准方式有点问题，pitch和incline角应该理论上是不能直接相加的，当相差的角度很小的时候，可以这么近似

    //rotate_and_convert_depth_map_to_point_cloud(intrinsic, depth_map_filepath, point_cloud_filepath);

    return 0;
}