#include "lio_sam/utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:

    rclcpp::Subscription<lio_sam::msg::CloudInfo>::SharedPtr subLaserCloudInfo; // 雷达点云信息订阅器

    rclcpp::Publisher<lio_sam::msg::CloudInfo>::SharedPtr pubLaserCloudInfo;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCornerPoints; // 角点特征发布器
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSurfacePoints; // 平面点特征发布器

    pcl::PointCloud<PointType>::Ptr extractedCloud; 
    pcl::PointCloud<PointType>::Ptr cornerCloud; // 角点点云
    pcl::PointCloud<PointType>::Ptr surfaceCloud; // 平面点点云

    pcl::VoxelGrid<PointType> downSizeFilter; // 降采样滤波器(降低角点和平面点密度)

    lio_sam::msg::CloudInfo cloudInfo;
    std_msgs::msg::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness; // 点云平滑度缓存(每个元素包含点的曲率和索引)
    float *cloudCurvature;  // 点云中的曲率
    int *cloudNeighborPicked;
    int *cloudLabel;

    FeatureExtraction(const rclcpp::NodeOptions & options) :
        ParamServer("lio_sam_featureExtraction", options)
    {   // 订阅前面经过去畸变的点云信息
        subLaserCloudInfo = create_subscription<lio_sam::msg::CloudInfo>(
            "lio_sam/deskew/cloud_info", qos,
            std::bind(&FeatureExtraction::laserCloudInfoHandler, this, std::placeholders::_1));

        pubLaserCloudInfo = create_publisher<lio_sam::msg::CloudInfo>(
            "lio_sam/feature/cloud_info", qos); // 发布点云
        pubCornerPoints = create_publisher<sensor_msgs::msg::PointCloud2>(
            "lio_sam/feature/cloud_corner", 1);  // 发布边缘类型点云
        pubSurfacePoints = create_publisher<sensor_msgs::msg::PointCloud2>(
            "lio_sam/feature/cloud_surface", 1);  // 发布平面类型点云

        initializationValue();
    }

    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }

    void laserCloudInfoHandler(const lio_sam::msg::CloudInfo::SharedPtr msgIn)
    {
        cloudInfo = *msgIn; // new cloud info
        cloudHeader = msgIn->header; // new cloud header
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // 将去畸变的点云信息转换为pcl格式，用于后续处理

        calculateSmoothness(); // 计算点云中的曲率

        markOccludedPoints(); // 标记遮挡点和平行光束点，避免后面进行错误的特征提取

        extractFeatures(); // 提取点云特征

        publishFeatureCloud(); // 发布特征点信息
    }

    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++) // 计算每个点左右共十个点对此点的曲率(平滑度)
        {
            float diffRange = cloudInfo.point_range[i-5] + cloudInfo.point_range[i-4]
                            + cloudInfo.point_range[i-3] + cloudInfo.point_range[i-2]
                            + cloudInfo.point_range[i-1] - cloudInfo.point_range[i] * 10
                            + cloudInfo.point_range[i+1] + cloudInfo.point_range[i+2]
                            + cloudInfo.point_range[i+3] + cloudInfo.point_range[i+4]
                            + cloudInfo.point_range[i+5]; // 方差
            // 平方差(曲率的一种计算)
            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting
            // 缓存点的曲率和索引，方便后面对曲率进行排序
            cloudSmoothness[i].value = cloudCurvature[i]; // 平滑度
            cloudSmoothness[i].ind = i;  // 对应点索引
        }
    }
    /**
     * @brief 标记遮挡点和平行光束点
     */
    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        // 标记遮挡点和平行光束点
        // 给标记成遮挡点和平行光束点的点打上标签，不作为特征点
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            float depth1 = cloudInfo.point_range[i];
            float depth2 = cloudInfo.point_range[i+1];
            int columnDiff = std::abs(int(cloudInfo.point_col_ind[i+1] - cloudInfo.point_col_ind[i])); // 获取两个点的列索引差值
            if (columnDiff < 10){ // 如果列索引差较小，即两点在扫描角度上靠得很近
                // 10 pixel diff in range image
                // 如果深度值相差较大，则将相邻得几个点标记为遮挡点，并标记为已选择过，后面不会对这些点进行特征提取
                if (depth1 - depth2 > 0.3){ 
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // 获取当前点与左右相邻之间的深度差
            float diff1 = std::abs(float(cloudInfo.point_range[i-1] - cloudInfo.point_range[i]));
            float diff2 = std::abs(float(cloudInfo.point_range[i+1] - cloudInfo.point_range[i]));
            // 如果相邻深度差都较大，则认为当前点为平行光束点，标记为已选择过，后续不会对这些点进行特征提取
            if (diff1 > 0.02 * cloudInfo.point_range[i] && diff2 > 0.02 * cloudInfo.point_range[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        for (int i = 0; i < N_SCAN; i++) // 对多线激光每一线内点循环
        {
            surfaceCloudScan->clear();
            // 将每根线均分为六段，分别对每一段进行特征提取
            for (int j = 0; j < 6; j++)
            {
                // 计算每一线中每一段的开始点和结束点
                int sp = (cloudInfo.start_ring_index[i] * (6 - j) + cloudInfo.end_ring_index[i] * j) / 6;
                int ep = (cloudInfo.start_ring_index[i] * (5 - j) + cloudInfo.end_ring_index[i] * (j + 1)) / 6 - 1;

                if (sp >= ep) 
                    continue;
                // 基于by_value()对迭代器之间的数进行排序
                // 对每一段内平滑度排序，从小到大
                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--) // 曲率倒序索引，从大到小
                {
                    int ind = cloudSmoothness[k].ind; // 获取当前点检索点的索引
                    // 判断cloudNeighborPicked中是否标记为遮挡点，平滑度是否大于阈值
                    // 角点判断
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold) 
                    {
                        largestPickedNum++;
                        if (largestPickedNum <= 20){ // 每段最多提取20个角点
                            cloudLabel[ind] = 1;
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        } else {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1; // 标记此点已经被处理过了，已经标记为角点了
                        // 判角点前后各五个点是否与其相差过大，是则标记为已经处理过了，防止角点聚集
                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                // 每一段内判断平面点，从前往后循环，从曲率小往大判断
                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {

                        cloudLabel[ind] = -1; // 打上平面点的标签
                        cloudNeighborPicked[ind] = 1; // 表示点已经被处理过了
                        // 同理防止聚集
                        for (int l = 1; l <= 5; l++) {
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                // 根据标记获取平面点
                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0){
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }
            // 对平面点降采样
            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);
            // 保存降采样后的平面点
            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    void freeCloudInfoMemory()
    {
        cloudInfo.start_ring_index.clear();
        cloudInfo.end_ring_index.clear();
        cloudInfo.point_col_ind.clear();
        cloudInfo.point_range.clear();
    }

    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory(); // 释放内存
        // save newly extracted features
        cloudInfo.cloud_corner = publishCloud(pubCornerPoints,  cornerCloud,  cloudHeader.stamp, lidarFrame); // 保存并发布角点
        cloudInfo.cloud_surface = publishCloud(pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame); // 保存并发布平面点
        // publish to mapOptimization
        pubLaserCloudInfo->publish(cloudInfo); // 发布点云信息
    }
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::SingleThreadedExecutor exec;

    auto FE = std::make_shared<FeatureExtraction>(options);

    exec.add_node(FE);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Feature Extraction Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();
    return 0;
}
