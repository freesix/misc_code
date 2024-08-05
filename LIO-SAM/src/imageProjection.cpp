#include "lio_sam/utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"
/**
 * @brief Velodyne雷达的点云数据类型
 * @details 通过pcl库定义Velodyne雷达的点云数据类型，包括点的坐标、强度、环数、时间戳
 */
struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D   // 定义点的坐标
    PCL_ADD_INTENSITY; // 定义点的强度
    uint16_t ring;  // 环数
    float time;  // 时间戳
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)
/**
 * @brief Ouster雷达的点云数据类型
 * @details 通过pcl库定义Ouster雷达的点云数据类型，包括点的坐标、强度、环数、时间戳、反射率、噪声、距离
 */
struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity; // 反射率
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

// Use the Velodyne point format as a common representation
// 将VelodynePointXYZIRT作为通用的点云数据类型
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000; // imu数据队列长度

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock; // imu数据队列锁
    std::mutex odoLock; // imu里程计队列锁

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subLaserCloud; // 雷达消息订阅器
    rclcpp::CallbackGroup::SharedPtr callbackGroupLidar;  
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloud; 

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubExtractedCloud; // 发布一系列预处理后有效的点云数据
    rclcpp::Publisher<lio_sam::msg::CloudInfo>::SharedPtr pubLaserCloudInfo; // 发布点云信息

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu; // IMU消息订阅器
    rclcpp::CallbackGroup::SharedPtr callbackGroupImu;
    std::deque<sensor_msgs::msg::Imu> imuQueue;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdom; // 里程计消息订阅器
    rclcpp::CallbackGroup::SharedPtr callbackGroupOdom;
    std::deque<nav_msgs::msg::Odometry> odomQueue;

    std::deque<sensor_msgs::msg::PointCloud2> cloudQueue; // 原始雷达消息缓存队列
    sensor_msgs::msg::PointCloud2 currentCloudMsg; // 从点云队列中提取出当前点云帧
    // imu积分获得的姿态信息，用于进行点云畸变矫正(作用只有一帧雷达点云)
    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur; // 记录每一帧点云其中过程中的imu数据指针
    bool firstPointFlag; // 处理第一个时，将该点旋转取逆，记录到transStartInverse中，方便后续计算
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<PointType>::Ptr   fullCloud; // 完整点云
    pcl::PointCloud<PointType>::Ptr   extractedCloud; // 畸变矫正后点云

    int deskewFlag; // 当点云的time/t字段不可用，也就是不包含时间戳，无法进行去畸变，直接返回原始点云
    cv::Mat rangeMat; // 点云投影获得的深度图
    // 进行点云畸变矫正时需要通过imu里程计获得imu位置增量
    bool odomDeskewFlag; // 是否有合适的imu里程计数据
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lio_sam::msg::CloudInfo cloudInfo;
    double timeScanCur; // 雷达帧开始扫描时间
    double timeScanEnd; // 雷达帧结束扫描时间
    std_msgs::msg::Header cloudHeader;

    vector<int> columnIdnCountVec;


public:
    ImageProjection(const rclcpp::NodeOptions & options) :
            ParamServer("lio_sam_imageProjection", options), deskewFlag(0)
    {   
        /*
        这里区别于普通的监听器在于使用ROS2的callbackGroup来指定下面三个callback都是不可并行的MutuallyExclusive.
        MutuallyExclusive 保证了所有缓存队列都是时间有序的.
        PS:根据ROS2的文档，没有指定callbackGroup的回调函数会注册到默认的callbackGroup，
        默认的callbackGroup本身就是Mutually Exclusive CallbackGroups，但是这样多个callback就没法并行执行。
        期望不同的callback可以并行执行，但是相同的callback不并行，推荐的做法就是将不同的callback注册到不同的
        Mutually Exclusive CallbackGroups
        + 关于ROS2的执行器Executors： https://docs.ros.org/en/foxy/Concepts/About-Executors.html
        + Using Callback Groups: https://docs.ros.org/en/foxy/How-To-Guides/Using-callback-groups.html
        */
        // 我的理解就是将这三个callback注册到不同的callbackGroup中，保证并行。但同一个callbackGroup中的callback不并行
        callbackGroupLidar = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupImu = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupOdom = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        auto lidarOpt = rclcpp::SubscriptionOptions();
        lidarOpt.callback_group = callbackGroupLidar;
        auto imuOpt = rclcpp::SubscriptionOptions();
        imuOpt.callback_group = callbackGroupImu;
        auto odomOpt = rclcpp::SubscriptionOptions();
        odomOpt.callback_group = callbackGroupOdom;

        subImu = create_subscription<sensor_msgs::msg::Imu>(
            imuTopic, qos_imu,
            std::bind(&ImageProjection::imuHandler, this, std::placeholders::_1),
            imuOpt);
        subOdom = create_subscription<nav_msgs::msg::Odometry>(
            odomTopic + "_incremental", qos_imu,
            std::bind(&ImageProjection::odometryHandler, this, std::placeholders::_1),
            odomOpt);
        subLaserCloud = create_subscription<sensor_msgs::msg::PointCloud2>(
            pointCloudTopic, qos_lidar,
            std::bind(&ImageProjection::cloudHandler, this, std::placeholders::_1),
            lidarOpt);

        pubExtractedCloud = create_publisher<sensor_msgs::msg::PointCloud2>(
            "lio_sam/deskew/cloud_deskewed", 1);
        pubLaserCloudInfo = create_publisher<lio_sam::msg::CloudInfo>(
            "lio_sam/deskew/cloud_info", qos);
        // 为动态指针和动态数组分配内存，初始化指针，否则会报错
        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR); // 管理pcl日志输出等级
    }
    // 分配内存空间
    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN); // 最大点云数量

        cloudInfo.start_ring_index.assign(N_SCAN, 0);
        cloudInfo.end_ring_index.assign(N_SCAN, 0);

        cloudInfo.point_col_ind.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.point_range.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }
    /**
     * @brief 重置参数
     * @details 重置参数，处理完每帧lidar数据都要重置这些参数
    */
    void resetParameters()
    {
        laserCloudIn->clear(); // 雷达点云信息清空
        extractedCloud->clear(); // 畸变矫正后点云清空
        // reset range matrix for range image projection 
        // Horizon_SCAN横向旋转扫描了多少次
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX)); // 投影深度图

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;
        // 清空imu积分获得的姿态信息缓存队列
        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection(){}
    /**
     * @brief IMU消息处理函数
     */
    void imuHandler(const sensor_msgs::msg::Imu::SharedPtr imuMsg)
    {
        sensor_msgs::msg::Imu thisImu = imuConverter(*imuMsg); // 将imu消息转换为雷达坐标系下(仅旋转对齐)

        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu); // 缓存转换后的imu数据

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }
    /**
     * @brief 缓存imu里程计信息，来自imuPreintegration发布的imu里程计
     */
    void odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }
    /**
     * @brief 点云消息处理函数
     * @param 输入的雷达消息
     */
    void cloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg)
    {
        if (!cachePointCloud(laserCloudMsg)) // 提取、转换点云为统一格式
            return;

        if (!deskewInfo()) // 从imu数据和imu里程计数据中提取去畸变信息
            return;

        projectPointCloud(); // 当前帧激光点云去运动畸变矫正

        cloudExtraction(); // 提取有效激光点，并整备发布的cloud_info数据包

        publishClouds(); // 发布当前帧矫正后点云，有效点和其它信息

        resetParameters(); // 重置参数，接受每帧lidar数据都要重置这些参数
    }

    bool cachePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& laserCloudMsg)
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2) // 点云队列帧数小于2
            return false;

        // convert cloud
        // 从点云队列中获取最早的数据，使用std::move将对象转为右值引用，避免拷贝
        currentCloudMsg = std::move(cloudQueue.front()); // 获取队列第一个消息
        cloudQueue.pop_front(); // 弹出已经获取的消息
        // 判断sensor类型
        // 如果为一下两种则将ROS消息类型转换为pcl类型
        if (sensor == SensorType::VELODYNE || sensor == SensorType::LIVOX)
        {
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
        }
        else if (sensor == SensorType::OUSTER)
        {
            // Convert to Velodyne format 
            // 将OUSTER雷达ROS类型点云转换为Velodyne风格的pcl点云
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else // 否则报错提醒
        {
            RCLCPP_ERROR_STREAM(get_logger(), "Unknown sensor type: " << int(sensor));
            rclcpp::shutdown();
        }

        // get timestamp
        cloudHeader = currentCloudMsg.header; // 消息头包含这个消息生成时间(采集时间)，该数据的坐标系，序列号(验证数据是否丢失)
        timeScanCur = stamp2Sec(cloudHeader.stamp); //点云msg中header时间作为此帧开始时间
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time; // 结束时间

        // check dense flag
        // 判断数据是否稠密(包含NaN点云与否)，有则报错
        if (laserCloudIn->is_dense == false)
        {
            RCLCPP_ERROR(get_logger(), "Point cloud is not in dense format, please remove NaN points first!");
            rclcpp::shutdown();
        }

        // check ring channel
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring") // 点云信息是否包含ring值
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1) // 检查到有点云没有ring值
            {
                RCLCPP_ERROR(get_logger(), "Point cloud ring channel not available, please configure your point cloud data!");
                rclcpp::shutdown();
            }
        }

        // check point time
        // 检查点云点是否有时间戳，没有则不去畸变，直接用原始点云，但会导致系统漂移
        if (deskewFlag == 0) // 没有去畸变
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t")
                {   
                    deskewFlag = 1; // 当雷达点云信息内包含时间戳字段，则设置去畸变标志位为1，否则为-1不去畸变
                    break;
                }
            }
            if (deskewFlag == -1) // 检查是否有点云没有时间信息，若有点云没有时间信息则会给系统带来显著漂移
                RCLCPP_WARN(get_logger(), "Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }
    /**
     * @brief 从imu数据和imu里程计数据中提取去畸变信息
     * imu数据：
     *  1. 遍历当前激光帧起止时间范围内的imu数据，初始时刻对应imu的姿态角RPY设为当前帧的初始姿态角
     *  2. 用角速度、时间积分，计算每一时刻相对初始值的旋转量，初始时刻旋转为0
     * imu里程计数据：
     *  1. 遍历当前激光帧起止时间范围内的imu里程计数据，初始时刻对应imu里程计的位姿设为当前帧的初始位姿
     *  2. 用起始时刻对应的imu里程计数据，计算相对位姿变换，保存平移增量
    */
    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        // 如果IMU数据为空，IMU开始时间大于激光扫描时间，imu队列中最后一帧时间小于激光扫描结束时间都返回false
        // 保证了imu数据对当前帧起始时间内的全覆盖，因为去畸变必须依赖高频的imu数据
        if (imuQueue.empty() ||
            stamp2Sec(imuQueue.front().header.stamp) > timeScanCur ||
            stamp2Sec(imuQueue.back().header.stamp) < timeScanEnd)
        {
            RCLCPP_INFO(get_logger(), "Waiting for IMU data ...");
            return false;
        }
        // 通过积分获取当前雷达帧扫描开始和结束时间戳内的imu姿态信息
        imuDeskewInfo(); // 从imu队列中计算去畸变信息

        odomDeskewInfo(); // 从imu里程计中计算去畸变信息

        return true;
    }

    void imuDeskewInfo()
    {
        cloudInfo.imu_available = false;

        while (!imuQueue.empty())
        {
            // 如果IMU时间小于当前激光帧，则舍弃
            // 修剪IMU数据，直到imu的时间处于这帧点云内
            if (stamp2Sec(imuQueue.front().header.stamp) < timeScanCur - 0.01) 
                imuQueue.pop_front(); 
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;
        // 循环遍历已经截取后的IMU数据
        // 所以下面的if语句能保证落入条件，不存在不符合的条件
        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::msg::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = stamp2Sec(thisImuMsg.header.stamp);

            // get roll, pitch, and yaw estimation for this scan
            if (currentImuTime <= timeScanCur)
                // 将IMU的四元数表示的姿态信息转换为欧拉角表示的姿态信息
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imu_roll_init, &cloudInfo.imu_pitch_init, &cloudInfo.imu_yaw_init);
            if (currentImuTime > timeScanEnd + 0.01) // imu数据超过当前激光帧结束0.01s则结束循环遍历imu数据
                break;

            if (imuPointerCur == 0){ // 初始化第一帧imu姿态信息为0
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            // 获取IMU的角速度
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation
            // 积分角速度得到旋转角度
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;
        // 用imu数据去畸变成功
        cloudInfo.imu_available = true;
    }

    void odomDeskewInfo()
    {
        cloudInfo.odom_available = false;
        // 和前面一样，保证里程计数据在激光帧前0.01到激光最后一帧之间
        // 丢弃早于当前雷达帧开始扫描时间的imu里程计帧
        while (!odomQueue.empty())
        {
            if (stamp2Sec(odomQueue.front().header.stamp) < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;
        // 这和第一个条件形成让队列中第一帧imu在激光帧开始时间之前，但又不超过其0.01s
        if (stamp2Sec(odomQueue.front().header.stamp) > timeScanCur) 
            return;

        // get start odometry at the beinning of the scan
        nav_msgs::msg::Odometry startOdomMsg;
        // 获取到的里程计信息开始帧时间大于等于激光当前帧开始时间
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (stamp2Sec(startOdomMsg.header.stamp) < timeScanCur)
                continue;
            else
                break;
        }

        tf2::Quaternion orientation; // 定义一个四元数 表示旋转
        // 将消息队列里的四元数转换为定义的四元数
        tf2::fromMsg(startOdomMsg.pose.pose.orientation, orientation);
        // 获取欧拉角表示的形式
        double roll, pitch, yaw;
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization
        // 用当前激光帧起始时刻的imu里程计 初始化为雷达的初始位姿，用于后面的图优化
        cloudInfo.initial_guess_x = startOdomMsg.pose.pose.position.x;
        cloudInfo.initial_guess_y = startOdomMsg.pose.pose.position.y;
        cloudInfo.initial_guess_z = startOdomMsg.pose.pose.position.z;
        cloudInfo.initial_guess_roll = roll;
        cloudInfo.initial_guess_pitch = pitch;
        cloudInfo.initial_guess_yaw = yaw;

        cloudInfo.odom_available = true;

        // get end odometry at the end of the scan
        // 是否使用IMU里程计数据对点云去畸变的标志位
        // 前面找到了点云起始时刻的IMU里程计，只有在同时找到结束时刻的IMU里程计数据，才可以
        // 利用时间插值算出每一个点对应时刻的位姿做去畸变
        // 注意：！在官方代码中，最后并没有使用IMU里程计数据做去畸变。所以这个标志位世纪没有被使用，
        // 下面的这些代码实际上也没有被使用到
        // 为什么没有使用IMU里程计做去畸变处理？
        // - 原代码中的注释写的是速度较低的情况下不需要做平移去畸变
        odomDeskewFlag = false;

        if (stamp2Sec(odomQueue.back().header.stamp) < timeScanEnd)
            return;

        nav_msgs::msg::Odometry endOdomMsg;
        // 找到第一帧时间大于激光结束时间的里程计IMU数据数据
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (stamp2Sec(endOdomMsg.header.stamp) < timeScanEnd)
                continue;
            else
                break;
        }
        // 比较起始帧之间位姿信息的协方差矩阵
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;
        // 根据提供的初始位姿和一个欧拉角计算仿射变换矩阵(imu里程计的起始帧位姿)
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf2::fromMsg(endOdomMsg.pose.pose.orientation, orientation); 
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw); // 将orientation姿态信息转换为roll、pitch、yaw欧拉角信息
        // 同上，不过是endOdomMsg携带的变换信息
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        Eigen::Affine3f transBt = transBegin.inverse() * transEnd; // 获取imu结束帧相对于开始帧的变换矩阵(在雷达帧扫描时间内的imu)

        float rollIncre, pitchIncre, yawIncre;
        // 这句代码是从仿射变换矩阵中获得位姿的增量变化，包括位移和欧拉角增量
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);
        // 设置里程计去畸变标志为true，标志后面可以用imu里程计数据做去畸变
        odomDeskewFlag = true;
    }
    /**
     * @brief 在当前激光帧起止时间范围内，计算某一时刻的旋转
     * @param pointTime 要计算的某一时刻
     */
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        int imuPointerFront = 0;
        // 查找第一个时间戳大于等于当前点的imu数据指针
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront]) // 如果pointTime小于imuPointerFront帧数对应的imu时间
                break;                                // 意思就是找到刚好大于pointTime的那一帧imu
            ++imuPointerFront;
        }
        // 这个if条件可能是最后一个imu时间戳依然小于雷达点的时间戳 
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0) // 这里就是遍历完imu时间，都还没到pointTime 直接返回最近的
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {  // 这里是找到了刚好大于pointTime的那一帧，则加权插值
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }
    /**
     * @brief 根据某一个点的时间戳从IMU里程计去畸变信息列表中找到对应的平移。
     * 这个函数根据时间间隔对平移量进行插值得到起止时间段之间任意时刻的位移
     * 
     * @param relTime 点云中某一个点的time字段（相对于起始点的时间）
     * @param poseXCur 输出的X轴方向平移量
     * @param poseYCur 输出的Y轴方向平移量
     * @param poseZCur 输出的Z轴方向平移量
    */
    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {   // 初始化为0
        *posXCur = 0; *posYCur = 0; *posZCur = 0;
        // 下面全注释掉是因为作者认为在慢速运动下，不需要做平移去畸变，也就是前面的用imu里程计数据做去畸变的标志位在下面没有被使用
        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanEnd - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }
    /**
     * @brief 激光运动畸变矫正、利用当前帧起止时刻之间的imu数据计算旋转增量
     * imu里程计数据计算平移增量、进而将每一时刻激光点位置变换到第一个激光点坐标系下面，进行运动补偿
     *  1.找到该点对应的旋转
     *  2.找到该点对应的平移
     *  3.坐标变换
     * @param point 点云中的一个点
     * @param relTime 点云中的一个点的time字段（相对于起始点的时间）
    */
    PointType deskewPoint(PointType *point, double relTime)
    {
        // 如果标志位-1或者imu数据没有做处理
        if (deskewFlag == -1 || cloudInfo.imu_available == false) 
            return *point;
        // 点的时间等于扫描帧开始时间+realtime(后文的laserClouIn->points[i].time)
        double pointTime = timeScanCur + relTime; 
        // 寻找某一时刻旋转增量(相对于帧扫描起始时刻)
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);
        // 这里的firstPointFlag来源于resetParameters函数，而resetParameters函数每次ros调用
        // cloudHandler都会启动
        if (firstPointFlag == true)
        {   // 计算此时刻变换矩阵的逆，意思是计算第一个点相对于世界坐标系的位姿
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start
        // 后续点相对于第一个点的坐标变换
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        // 第一个点相对于世界坐标系x当前点相对于世界坐标系变换=当前点相对于第一个点在世界坐标系的变换
        Eigen::Affine3f transBt = transStartInverse * transFinal;
        // 这里用位姿变换乘上每个点的坐标，进行每个点的坐标变换去畸变
        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }
    /**
     * @brief 执行点云去畸变
     * @details 1. 检查激光点距离，扫描线是否合规
     *          2. 单点去畸变 
     *          3. 保存激光点
    */
    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            float range = pointDistance(thisPoint); // 计算当前点的深度
            // 如果距离小于雷达最小距离或者大于最大距离，忽略此点
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;

            int rowIdn = laserCloudIn->points[i].ring; // 获取当前点所在的线索引，即在深度图中的行索引
            if (rowIdn < 0 || rowIdn >= N_SCAN) // 如果在给定的扫描线范围之外，忽略此点
                continue;

            if (rowIdn % downsampleRate != 0) // 降采样
                continue;

            int columnIdn = -1;
            if (sensor == SensorType::VELODYNE || sensor == SensorType::OUSTER)
            {   // 下面将点分配到具体某个columnIdn
                float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI; // 计算当前点的水平倾斜角
                static float ang_res_x = 360.0/float(Horizon_SCAN); // 计算水平角分辨率
                columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2; // 计算当前点在深度图中的列索引
                if (columnIdn >= Horizon_SCAN) // 大于Horizon_SCAN，超过了360度
                    columnIdn -= Horizon_SCAN; // 减去一周水平扫描帧数
            }
            else if (sensor == SensorType::LIVOX) // ?? 这个雷达作者用的应该不是环形扫描的雷达
            {
                columnIdn = columnIdnCountVec[rowIdn];
                columnIdnCountVec[rowIdn] += 1;
            }


            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;
            // 不在多线雷达扫描的点阵中
            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;
            // 去畸变、运动补偿
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);
            // 将处理后的点的距离值存入距离矩阵
            rangeMat.at<float>(rowIdn, columnIdn) = range;
            // 按照一定索引方式存储点
            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }
    /**
     * @brief 提取点云，并不是所有处理好的点都放进cloudInfo，而是剔除前五个点和后五个点
     * 不是真正的删除，而是将start_ring_index索引前移5个，end_ring_index索引后移5个
     * 因为计算曲率的时候没法计算前五个点和后五个点的曲率
     */
    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        // 确定每根线的起始和结束点索引，提出畸变矫正后的点云
        for (int i = 0; i < N_SCAN; ++i)
        {
            cloudInfo.start_ring_index[i] = count - 1 + 5;
            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    cloudInfo.point_col_ind[count] = j;
                    // save range info
                    cloudInfo.point_range[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.end_ring_index[i] = count -1 - 5;
        }
    }
    
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed  = publishCloud(pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo->publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor exec;

    auto IP = std::make_shared<ImageProjection>(options);
    exec.add_node(IP);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Image Projection Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();
    return 0;
}
