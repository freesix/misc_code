#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <iostream>
#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <opencv2/opencv.hpp>

#include <pcl/kdtree/kdtree_flann.h>  // pcl include kdtree_flann throws error if PCL_NO_PRECOMPILE
                                      // is defined before
#define PCL_NO_PRECOMPILE
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

using namespace std;

typedef pcl::PointXYZI PointType;

enum class SensorType { VELODYNE, OUSTER, LIVOX };

class ParamServer : public rclcpp::Node
{
public:
    std::string robot_id;

    //Topics
    string pointCloudTopic; // 原始点云数据话题
    string imuTopic; // 原始imu数据话题
    string odomTopic; // imu里程计话题，在imuPreintegration中对imu预积分得到
    string gpsTopic; // 原始gps经过robot_localization包计算得到

    //Frames
    string lidarFrame; // 激光雷达坐标系。点云数据坐标系，与lidarFarme一致，但不同雷达名称不同
    string baselinkFrame; // 机器人底盘坐标系
    string odometryFrame; // 里程计坐标系
    string mapFrame; // 地图坐标系，全局坐标系(如果odometryFrame和其第一帧对齐，那两者相同)

    // GPS Settings
    bool useImuHeadingInitialization;
    bool useGpsElevation;
    float gpsCovThreshold;
    float poseCovThreshold;

    // Save pcd
    bool savePCD;
    string savePCDDirectory; // 保存地图pcd文件的路径

    // Lidar Sensor Configuration
    SensorType sensor = SensorType::OUSTER;
    int N_SCAN;
    int Horizon_SCAN;
    int downsampleRate;
    float lidarMinRange;
    float lidarMaxRange;

    // IMU
    float imuAccNoise; // IMU加速度噪声协方差，可以用Allen方差标定，这里设三轴相同
    float imuGyrNoise; // IMU角速度噪声协方差，可以用Allen方差标定，这里设三轴相同
    float imuAccBiasN; // IMU加速度偏差，三轴统一
    float imuGyrBiasN; // IMU角速度偏差，三轴统一
    float imuGravity; // 重力加速度值
    float imuRPYWeight; // 算法中使用IMU的roll pitch角对激光里程计的结果加权融合
    vector<double> extRotV;  // IMU加速度向量到雷达坐标系的旋转
    vector<double> extRPYV; // IMU角速度向量到雷达坐标系的旋转
    vector<double> extTransV; // IMU到雷达坐标系的平移 
    Eigen::Matrix3d extRot; // IMU加速度向量到雷达坐标系的旋转
    Eigen::Matrix3d extRPY; // IMU角速度向量到雷达坐标系的旋转
    Eigen::Vector3d extTrans; // IMU到雷达坐标系的平移
    Eigen::Quaterniond extQRPY; // IMU角速度向量到雷达坐标系的旋转(四元数形式)

    // LOAM
    float edgeThreshold; // 边缘特征点阈值
    float surfThreshold; // 平面特征点阈值
    int edgeFeatureMinValidNum; // 边缘特征点最小数量
    int surfFeatureMinValidNum; // 平面特征点最小数量

    // voxel filter paprams
    float odometrySurfLeafSize; 
    float mappingCornerLeafSize;
    float mappingSurfLeafSize ;

    float z_tollerance; // 限制z轴平移的大小
    float rotation_tollerance; // 限制roll、pitch角的大小

    // CPU Params
    int numberOfCores; // 在点云匹配中使用的CPU核心数量
    double mappingProcessInterval; // 点云帧处理时间间隔

    // Surrounding map
    float surroundingkeyframeAddingDistThreshold; // 当前帧与上一帧距离大于此阈值才能作为关键帧
    float surroundingkeyframeAddingAngleThreshold; // 当前帧与上一帧角度大于此阈值才能作为关键帧
    float surroundingKeyframeDensity; // 构建局部地图对采用的关键帧做降采样
    float surroundingKeyframeSearchRadius; // 构建局部地图时关键帧的检索半径

    // Loop closure
    bool  loopClosureEnableFlag; // 回环检测线程的执行标志
    float loopClosureFrequency;  // 回环检测线程执行的频率
    int   surroundingKeyframeSize; // 回环检测构建局部地图的最大关键帧数量
    float historyKeyframeSearchRadius; // 回环检测时关键帧的检索半径
    float historyKeyframeSearchTimeDiff; // 回环检测时关键帧检索的时间范围
    int   historyKeyframeSearchNum; // 执行回环检测时融合局部地图对目标关键帧执行+-25帧的关键帧融合
    float historyKeyframeFitnessScore; // 执行回环检测时用ICP做点云匹配，阈值大于0.3认为匹配失败

    // global map visualization radius
    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;

    ParamServer(std::string node_name, const rclcpp::NodeOptions & options) : Node(node_name, options)
    {
        declare_parameter("pointCloudTopic", "points");
        get_parameter("pointCloudTopic", pointCloudTopic);
        declare_parameter("imuTopic", "imu/data");
        get_parameter("imuTopic", imuTopic);
        declare_parameter("odomTopic", "lio_sam/odometry/imu");
        get_parameter("odomTopic", odomTopic);
        declare_parameter("gpsTopic", "lio_sam/odometry/gps");
        get_parameter("gpsTopic", gpsTopic);

        declare_parameter("lidarFrame", "laser_data_frame");
        get_parameter("lidarFrame", lidarFrame);
        declare_parameter("baselinkFrame", "base_link");
        get_parameter("baselinkFrame", baselinkFrame);
        declare_parameter("odometryFrame", "odom");
        get_parameter("odometryFrame", odometryFrame);
        declare_parameter("mapFrame", "map");
        get_parameter("mapFrame", mapFrame);

        declare_parameter("useImuHeadingInitialization", false);
        get_parameter("useImuHeadingInitialization", useImuHeadingInitialization);
        declare_parameter("useGpsElevation", false);
        get_parameter("useGpsElevation", useGpsElevation);
        declare_parameter("gpsCovThreshold", 2.0);
        get_parameter("gpsCovThreshold", gpsCovThreshold);
        declare_parameter("poseCovThreshold", 25.0);
        get_parameter("poseCovThreshold", poseCovThreshold);

        declare_parameter("savePCD", false);
        get_parameter("savePCD", savePCD);
        declare_parameter("savePCDDirectory", "/Downloads/LOAM/");
        get_parameter("savePCDDirectory", savePCDDirectory);

        std::string sensorStr;
        declare_parameter("sensor", "ouster");
        get_parameter("sensor", sensorStr);
        if (sensorStr == "velodyne")
        {
            sensor = SensorType::VELODYNE;
        }
        else if (sensorStr == "ouster")
        {
            sensor = SensorType::OUSTER;
        }
        else if (sensorStr == "livox")
        {
            sensor = SensorType::LIVOX;
        }
        else
        {
            RCLCPP_ERROR_STREAM(
                get_logger(),
                "Invalid sensor type (must be either 'velodyne' or 'ouster' or 'livox'): " << sensorStr);
            rclcpp::shutdown();
        }

        declare_parameter("N_SCAN", 64);
        get_parameter("N_SCAN", N_SCAN);
        declare_parameter("Horizon_SCAN", 512);
        get_parameter("Horizon_SCAN", Horizon_SCAN);
        declare_parameter("downsampleRate", 1);
        get_parameter("downsampleRate", downsampleRate);
        declare_parameter("lidarMinRange", 5.5);
        get_parameter("lidarMinRange", lidarMinRange);
        declare_parameter("lidarMaxRange", 1000.0);
        get_parameter("lidarMaxRange", lidarMaxRange);

        declare_parameter("imuAccNoise", 9e-4);
        get_parameter("imuAccNoise", imuAccNoise);
        declare_parameter("imuGyrNoise", 1.6e-4);
        get_parameter("imuGyrNoise", imuGyrNoise);
        declare_parameter("imuAccBiasN", 5e-4);
        get_parameter("imuAccBiasN", imuAccBiasN);
        declare_parameter("imuGyrBiasN", 7e-5);
        get_parameter("imuGyrBiasN", imuGyrBiasN);
        declare_parameter("imuGravity", 9.80511);
        get_parameter("imuGravity", imuGravity);
        declare_parameter("imuRPYWeight", 0.01);
        get_parameter("imuRPYWeight", imuRPYWeight);

        double ida[] = { 1.0,  0.0,  0.0,
                         0.0,  1.0,  0.0,
                         0.0,  0.0,  1.0};
        std::vector < double > id(ida, std::end(ida));
        declare_parameter("extrinsicRot", id);
        get_parameter("extrinsicRot", extRotV);
        declare_parameter("extrinsicRPY", id);
        get_parameter("extrinsicRPY", extRPYV);
        double zea[] = {0.0, 0.0, 0.0};
        std::vector < double > ze(zea, std::end(zea));
        declare_parameter("extrinsicTrans", ze);
        get_parameter("extrinsicTrans", extTransV);

        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
        extQRPY = Eigen::Quaterniond(extRPY);

        declare_parameter("edgeThreshold", 1.0);
        get_parameter("edgeThreshold", edgeThreshold);
        declare_parameter("surfThreshold", 0.1);
        get_parameter("surfThreshold", surfThreshold);
        declare_parameter("edgeFeatureMinValidNum", 10);
        get_parameter("edgeFeatureMinValidNum", edgeFeatureMinValidNum);
        declare_parameter("surfFeatureMinValidNum", 100);
        get_parameter("surfFeatureMinValidNum", surfFeatureMinValidNum);

        declare_parameter("odometrySurfLeafSize", 0.4);
        get_parameter("odometrySurfLeafSize", odometrySurfLeafSize);
        declare_parameter("mappingCornerLeafSize", 0.2);
        get_parameter("mappingCornerLeafSize", mappingCornerLeafSize);
        declare_parameter("mappingSurfLeafSize", 0.4);
        get_parameter("mappingSurfLeafSize", mappingSurfLeafSize);

        declare_parameter("z_tollerance", 1000.0);
        get_parameter("z_tollerance", z_tollerance);
        declare_parameter("rotation_tollerance", 1000.0);
        get_parameter("rotation_tollerance", rotation_tollerance);

        declare_parameter("numberOfCores", 4);
        get_parameter("numberOfCores", numberOfCores);
        declare_parameter("mappingProcessInterval", 0.15);
        get_parameter("mappingProcessInterval", mappingProcessInterval);

        declare_parameter("surroundingkeyframeAddingDistThreshold", 1.0);
        get_parameter("surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold);
        declare_parameter("surroundingkeyframeAddingAngleThreshold", 0.2);
        get_parameter("surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold);
        declare_parameter("surroundingKeyframeDensity", 2.0);
        get_parameter("surroundingKeyframeDensity", surroundingKeyframeDensity);
        declare_parameter("surroundingKeyframeSearchRadius", 50.0);
        get_parameter("surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius);

        declare_parameter("loopClosureEnableFlag", true);
        get_parameter("loopClosureEnableFlag", loopClosureEnableFlag);
        declare_parameter("loopClosureFrequency", 1.0);
        get_parameter("loopClosureFrequency", loopClosureFrequency);
        declare_parameter("surroundingKeyframeSize", 50);
        get_parameter("surroundingKeyframeSize", surroundingKeyframeSize);
        declare_parameter("historyKeyframeSearchRadius", 15.0);
        get_parameter("historyKeyframeSearchRadius", historyKeyframeSearchRadius);
        declare_parameter("historyKeyframeSearchTimeDiff", 30.0);
        get_parameter("historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff);
        declare_parameter("historyKeyframeSearchNum", 25);
        get_parameter("historyKeyframeSearchNum", historyKeyframeSearchNum);
        declare_parameter("historyKeyframeFitnessScore", 0.3);
        get_parameter("historyKeyframeFitnessScore", historyKeyframeFitnessScore);

        declare_parameter("globalMapVisualizationSearchRadius", 1000.0);
        get_parameter("globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius);
        declare_parameter("globalMapVisualizationPoseDensity", 10.0);
        get_parameter("globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity);
        declare_parameter("globalMapVisualizationLeafSize", 1.0);
        get_parameter("globalMapVisualizationLeafSize", globalMapVisualizationLeafSize);

        usleep(100);
    }
    /**
     * @brief 将imu数据转换到雷达坐标系下，只做了旋转，没有平移
     * @param imu_in 输入的imu数据
     * @return sensor_msgs::msg::Imu 坐标转换后的imu数据
     */
    sensor_msgs::msg::Imu imuConverter(const sensor_msgs::msg::Imu& imu_in)
    {
        sensor_msgs::msg::Imu imu_out = imu_in;
        // rotate acceleration
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        acc = extRot * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();
        // rotate gyroscope
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
        gyr = extRot * gyr;
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();
        // rotate roll pitch yaw
        Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
        Eigen::Quaterniond q_final = q_from * extQRPY; // IMU位姿*IMU到雷达坐标系的旋转
        imu_out.orientation.x = q_final.x();
        imu_out.orientation.y = q_final.y();
        imu_out.orientation.z = q_final.z();
        imu_out.orientation.w = q_final.w();

        if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
        {
            RCLCPP_ERROR(get_logger(), "Invalid quaternion, please use a 9-axis IMU!");
            rclcpp::shutdown();
        }

        return imu_out;
    }
};


sensor_msgs::msg::PointCloud2 publishCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, rclcpp::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::msg::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->get_subscription_count() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

template<typename T>
double stamp2Sec(const T& stamp)
{
    return rclcpp::Time(stamp).seconds();
}


template<typename T>
void imuAngular2rosAngular(sensor_msgs::msg::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
{
    *angular_x = thisImuMsg->angular_velocity.x;
    *angular_y = thisImuMsg->angular_velocity.y;
    *angular_z = thisImuMsg->angular_velocity.z;
}


template<typename T>
void imuAccel2rosAccel(sensor_msgs::msg::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z)
{
    *acc_x = thisImuMsg->linear_acceleration.x;
    *acc_y = thisImuMsg->linear_acceleration.y;
    *acc_z = thisImuMsg->linear_acceleration.z;
}


template<typename T>
void imuRPY2rosRPY(sensor_msgs::msg::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
    double imuRoll, imuPitch, imuYaw;
    tf2::Quaternion orientation;
    tf2::fromMsg(thisImuMsg->orientation, orientation);
    tf2::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

    *rosRoll = imuRoll;
    *rosPitch = imuPitch;
    *rosYaw = imuYaw;
}


float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}
/**
 * @details 本算法中默认的QoS，保留最新的1个消息，不保证消息的可靠性，消息的持久性为瞬时，
 * 消息的生命周期为默认值，消息的活跃度为系统默认值，消息的活跃度持续时间为默认值，消息的历史性为系统默认值
*/
rmw_qos_profile_t qos_profile{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    1,
    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

auto qos = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile.history,
        qos_profile.depth
    ),
    qos_profile);
/**
 * @details imu 原始数据，因为imu数据小且频率高，所以depth设为2000，缓存中保留最新的2000
 * 个消息
*/
rmw_qos_profile_t qos_profile_imu{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    2000,
    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT, // 不保证每个消息到达接收端，但保证实时性
    RMW_QOS_POLICY_DURABILITY_VOLATILE, 
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

auto qos_imu = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile_imu.history,
        qos_profile_imu.depth
    ),
    qos_profile_imu);
/**
 * @details lidar原始数据topic的QoS
*/
rmw_qos_profile_t qos_profile_lidar{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    5,
    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

auto qos_lidar = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile_lidar.history,
        qos_profile_lidar.depth
    ),
    qos_profile_lidar);

#endif
