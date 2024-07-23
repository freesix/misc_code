#include "lio_sam/utility.hpp"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

class TransformFusion : public ParamServer
{
public:
    std::mutex mtx;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subImuOdometry; // 通过imu预积分估计的雷达里程计信息订阅器
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subLaserOdometry; // 最终优化后的里程计信息订阅器

    rclcpp::CallbackGroup::SharedPtr callbackGroupImuOdometry;
    rclcpp::CallbackGroup::SharedPtr callbackGroupLaserOdometry;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry; // imu里程计信息发布器
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubImuPath; // imu路径发布器

    Eigen::Isometry3d lidarOdomAffine;
    Eigen::Isometry3d imuOdomAffineFront;
    Eigen::Isometry3d imuOdomAffineBack;

    std::shared_ptr<tf2_ros::Buffer> tfBuffer;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster;
    std::shared_ptr<tf2_ros::TransformListener> tfListener;
    tf2::Stamped<tf2::Transform> lidar2Baselink;

    double lidarOdomTime = -1;
    deque<nav_msgs::msg::Odometry> imuOdomQueue;

    TransformFusion(const rclcpp::NodeOptions & options) : ParamServer("lio_sam_transformFusion", options)
    {   // get_clock()返回的时钟对象传递给构造函数，tfBuffer将使用ros2中的时钟来处理时间信息
        tfBuffer = std::make_shared<tf2_ros::Buffer>(get_clock());
        tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer);
        // rclcpp::CallbackGroupType::MutuallyExclusive表示回调组类型是互斥的
        // 同一时刻只能有一个回调函数在该组中被执行
        callbackGroupImuOdometry = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupLaserOdometry = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        auto imuOdomOpt = rclcpp::SubscriptionOptions();
        imuOdomOpt.callback_group = callbackGroupImuOdometry;
        auto laserOdomOpt = rclcpp::SubscriptionOptions();
        laserOdomOpt.callback_group = callbackGroupLaserOdometry;

        subLaserOdometry = create_subscription<nav_msgs::msg::Odometry>(
            "lio_sam/mapping/odometry", qos,
            std::bind(&TransformFusion::lidarOdometryHandler, this, std::placeholders::_1),
            laserOdomOpt);
        subImuOdometry = create_subscription<nav_msgs::msg::Odometry>(
            odomTopic+"_incremental", qos_imu,
            std::bind(&TransformFusion::imuOdometryHandler, this, std::placeholders::_1),
            imuOdomOpt);

        pubImuOdometry = create_publisher<nav_msgs::msg::Odometry>(odomTopic, qos_imu);
        pubImuPath = create_publisher<nav_msgs::msg::Path>("lio_sam/imu/path", qos);

        tfBroadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    }
    /**
     * @brief 将nav_msgs::msg::Odometry类型的消息转换为Eigen::Isometry3d类型的变换矩阵
     */
    Eigen::Isometry3d odom2affine(nav_msgs::msg::Odometry odom)
    {
        tf2::Transform t;
        tf2::fromMsg(odom.pose.pose, t);
        return tf2::transformToEigen(tf2::toMsg(t));
    }

    void lidarOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        lidarOdomAffine = odom2affine(*odomMsg);

        lidarOdomTime = stamp2Sec(odomMsg->header.stamp);
    }

    void imuOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        imuOdomQueue.push_back(*odomMsg); // 记录通过imu估计的雷达里程计信息

        // get latest odometry (at current IMU stamp)
        if (lidarOdomTime == -1) // 没有订阅到的最终优化后的里程计信息，返回
            return;
        while (!imuOdomQueue.empty()) // 删除比该帧激光里程计时间戳更早的imu里程计信息
        {
            if (stamp2Sec(imuOdomQueue.front().header.stamp) <= lidarOdomTime)
                imuOdomQueue.pop_front();
            else
                break;
        }
        Eigen::Isometry3d imuOdomAffineFront = odom2affine(imuOdomQueue.front()); // 获取最早的imu里程计信息
        Eigen::Isometry3d imuOdomAffineBack = odom2affine(imuOdomQueue.back()); // 获取最新的imu里程计信息
        Eigen::Isometry3d imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack; // 获取新老帧之间的位姿增量
        Eigen::Isometry3d imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre; // 雷达里程计信息加上imu里程计信息增量
        auto t = tf2::eigenToTransform(imuOdomAffineLast);
        tf2::Stamped<tf2::Transform> tCur;
        tf2::convert(t, tCur);

        // publish latest odometry
        nav_msgs::msg::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = t.transform.translation.x;
        laserOdometry.pose.pose.position.y = t.transform.translation.y;
        laserOdometry.pose.pose.position.z = t.transform.translation.z;
        laserOdometry.pose.pose.orientation = t.transform.rotation;
        pubImuOdometry->publish(laserOdometry);

        // publish tf
        if(lidarFrame != baselinkFrame)
        {
            try
            {
                tf2::fromMsg(tfBuffer->lookupTransform(
                    lidarFrame, baselinkFrame, rclcpp::Time(0)), lidar2Baselink);
            }
            catch (tf2::TransformException ex)
            {
                RCLCPP_ERROR(get_logger(), "%s", ex.what());
            }
            tf2::Stamped<tf2::Transform> tb(
                tCur * lidar2Baselink, tf2_ros::fromMsg(odomMsg->header.stamp), odometryFrame);
            tCur = tb;
        }
        geometry_msgs::msg::TransformStamped ts;
        tf2::convert(tCur, ts);
        ts.child_frame_id = baselinkFrame;
        tfBroadcaster->sendTransform(ts);

        // publish IMU path
        static nav_msgs::msg::Path imuPath;
        static double last_path_time = -1;
        double imuTime = stamp2Sec(imuOdomQueue.back().header.stamp);
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            imuPath.poses.push_back(pose_stamped);
            while(!imuPath.poses.empty() && stamp2Sec(imuPath.poses.front().header.stamp) < lidarOdomTime - 1.0)
                imuPath.poses.erase(imuPath.poses.begin());
            if (pubImuPath->get_subscription_count() != 0)
            {
                imuPath.header.stamp = imuOdomQueue.back().header.stamp;
                imuPath.header.frame_id = odometryFrame;
                pubImuPath->publish(imuPath);
            }
        }
    }
};

class IMUPreintegration : public ParamServer
{
public:

    std::mutex mtx;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdometry;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry;

    rclcpp::CallbackGroup::SharedPtr callbackGroupImu;
    rclcpp::CallbackGroup::SharedPtr callbackGroupOdom;

    bool systemInitialized = false;

    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
    gtsam::Vector noiseModelBetweenBias;


    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    std::deque<sensor_msgs::msg::Imu> imuQueOpt;
    std::deque<sensor_msgs::msg::Imu> imuQueImu;

    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;

    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;
    gtsam::Values graphValues;

    const double delta_t = 0;

    int key = 1;

    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

    IMUPreintegration(const rclcpp::NodeOptions & options) :
            ParamServer("lio_sam_imu_preintegration", options)
    {
        callbackGroupImu = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive); // 创建一个回调组
        callbackGroupOdom = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        auto imuOpt = rclcpp::SubscriptionOptions(); // 定义可配置的订阅者类
        imuOpt.callback_group = callbackGroupImu;
        auto odomOpt = rclcpp::SubscriptionOptions();
        odomOpt.callback_group = callbackGroupOdom;

        subImu = create_subscription<sensor_msgs::msg::Imu>(
            imuTopic, qos_imu,  // qos_imu类用于控制消息传递的服务质量
            std::bind(&IMUPreintegration::imuHandler, this, std::placeholders::_1),
            imuOpt);
        subOdometry = create_subscription<nav_msgs::msg::Odometry>(
            "lio_sam/mapping/odometry_incremental", qos,
            std::bind(&IMUPreintegration::odometryHandler, this, std::placeholders::_1),
            odomOpt);

        pubImuOdometry = create_publisher<nav_msgs::msg::Odometry>(odomTopic+"_incremental", qos_imu);
        // 定义一个gtsam库中的预计分对象指针，并初始化了重力方向。
        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous
        p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
        p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias
        // 先验位置噪声
        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        // 先验速度噪声
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // m/s
        // 先验偏置噪声
        priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // rad,rad,rad,m, m, m
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
        
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization        
    }
    /**
     * @brief 重置优化器
     */
    void resetOptimization()
    {   // 使用指定参数重新初始化优化器
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);
        // 清空当前的因子图
        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;
        // 清空当前的变量值
        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }
    /**
     * @brief 重置参数
     */
    void resetParams()
    {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }
    /**
     * @brief 处理来自mapOptmization的激光里程计信息(相对世界坐标系的位姿)
     * 订阅的是“lio_sam/mapping/odometry_incremental”话题
    */
    void odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        double currentCorrectionTime = stamp2Sec(odomMsg->header.stamp); // 当前激光里程计时间戳

        // make sure we have imu data to integrate
        if (imuQueOpt.empty()) // 确保imu优化队列中有imu数据进行积分
            return;

        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;  // 旋转信息，四元数方式存储
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));


        // 0. initialize system
        // 矫正过程的初始化
        if (systemInitialized == false)
        {
            resetOptimization(); // 重置ISAM2优化器

            // pop old IMU message
            // 丢弃老的imu信息(小于当前激光里程计时间戳的)
            while (!imuQueOpt.empty())
            {
                if (stamp2Sec(imuQueOpt.front().header.stamp) < currentCorrectionTime - delta_t)
                {
                    lastImuT_opt = stamp2Sec(imuQueOpt.front().header.stamp);
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }
            // initial pose
            prevPose_ = lidarPose.compose(lidar2Imu); // 把雷达里程计的位姿转换到“中间系”下
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise); // 位姿因子
            graphFactors.add(priorPose);
            // initial velocity
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            graphFactors.add(priorVel);
            // initial bias
            prevBias_ = gtsam::imuBias::ConstantBias(); // 偏置因子(没有初始值)
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            graphFactors.add(priorBias);
            // add values 变量节点赋初值
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            optimizer.update(graphFactors, graphValues); // 优化一次
            //图和节点均清零  为什么要清零不能继续用吗?
            //是因为节点信息保存在gtsam::ISAM2 optimizer，所以要清理后才能继续使用
            graphFactors.resize(0);
            graphValues.clear();
            // 积分器重置，重置优化之后的偏置
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            
            key = 1;
            systemInitialized = true;
            return;
        }


        // reset graph for speed 每隔100个imu数据，重置一次优化器，保证优化效率
        if (key == 100)
        {
            // get updated noise before reset 保存噪声值
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));
            // reset graph
            resetOptimization();
            // add pose
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // add velocity
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // add bias
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1;
        }


        // 1. integrate imu data and optimize
        // 计算前一帧和当前帧之间的imu预积分量，用前一帧状态施加预积分量得到当前帧初始估计状态
        // 添加来自mapOptimization的当前帧位姿，进行因子图优化，更新当前帧位姿
        while (!imuQueOpt.empty())
        {
            // pop and integrate imu data that is between two optimizations
            sensor_msgs::msg::Imu *thisImu = &imuQueOpt.front();
            double imuTime = stamp2Sec(thisImu->header.stamp);
            if (imuTime < currentCorrectionTime - delta_t)
            {
                double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                
                lastImuT_opt = imuTime;
                imuQueOpt.pop_front();
            }
            else
                break;
        }
        // add imu factor to graph
        const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        graphFactors.add(imu_factor);
        // add imu bias between factor
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
        // add pose factor
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
        graphFactors.add(pose_factor);
        // insert predicted values
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
        // optimize
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        graphFactors.resize(0);
        graphValues.clear();
        // Overwrite the beginning of the preintegration for the next step.
        gtsam::Values result = optimizer.calculateEstimate();
        prevPose_  = result.at<gtsam::Pose3>(X(key));
        prevVel_   = result.at<gtsam::Vector3>(V(key));
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));
        // Reset the optimization preintegration object.
        // 重置预积分器，设置新的偏差，这样下一帧激光里程计进来的时候，预积分量就是两帧之间的增量
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        // check optimization
        // imu因子图优化结果，速度或者偏置过大，则认为失败
        if (failureDetection(prevVel_, prevBias_))
        {
            resetParams();
            return;
        }


        // 2. after optiization, re-propagate imu odometry preintegration
        // 优化之后，重新传播，优化更新了imu的偏置
        // 用最新的偏置重新计算当前激光例程计时刻之后的imu预积分，这个预积分用于计算每时刻位姿
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;
        // first pop imu message older than current correction data
        double lastImuQT = -1;
        while (!imuQueImu.empty() && stamp2Sec(imuQueImu.front().header.stamp) < currentCorrectionTime - delta_t)
        {
            lastImuQT = stamp2Sec(imuQueImu.front().header.stamp);
            imuQueImu.pop_front();
        }
        // repropogate
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // integrate imu message from the beginning of this optimization
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                sensor_msgs::msg::Imu *thisImu = &imuQueImu[i];
                double imuTime = stamp2Sec(thisImu->header.stamp);
                double dt = (lastImuQT < 0) ? (1.0 / 500.0) :(imuTime - lastImuQT);

                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }

        ++key;
        doneFirstOpt = true;
    }
    /**
     * @brief 检测积分条件是否合适，是否需要重新积分
     */
    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > 30) // 速度太大
        {
            RCLCPP_WARN(get_logger(), "Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 1.0 || bg.norm() > 1.0) // 偏置太大
        {
            RCLCPP_WARN(get_logger(), "Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }

    void imuHandler(const sensor_msgs::msg::Imu::SharedPtr imu_raw)
    {
        std::lock_guard<std::mutex> lock(mtx);
        // 将imu信息转换到雷达坐标系下表示(只转换了旋转，差了平移)
        sensor_msgs::msg::Imu thisImu = imuConverter(*imu_raw);

        imuQueOpt.push_back(thisImu); // 一个imu的中间缓存，用于优化imu的bias之类
        imuQueImu.push_back(thisImu); // 一直用到的imu数据

        if (doneFirstOpt == false)
            return;

        double imuTime = stamp2Sec(thisImu.header.stamp);
        // 获取两个imu数据之间的时间间隔，如果lastImuT_imu<0，说明是第一个imu数据，赋值为1/500
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu); 
        lastImuT_imu = imuTime; // 更新前一个imu数据时间

        // integrate this single imu message
        // 向imu积分器中添加一个imu数据
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

        // predict odometry
        // 利用上一时刻里程计的PVQ和偏置信息，预计分当前时刻imu里程计的PVQ
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry
        auto odometry = nav_msgs::msg::Odometry();
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = odometryFrame;
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar); // 把imuPose通过imu2Lidar从“中间系”个平移到到真正的雷达系
        // 发布通过imu估计的雷达里程计信息(也称imu里程计信息)
        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        
        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        pubImuOdometry->publish(odometry);
    }
};


int main(int argc, char** argv)
{   
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor e;

    auto ImuP = std::make_shared<IMUPreintegration>(options);
    auto TF = std::make_shared<TransformFusion>(options);
    e.add_node(ImuP);
    e.add_node(TF);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> IMU Preintegration Started.\033[0m");

    e.spin();

    rclcpp::shutdown();
    return 0;
}
