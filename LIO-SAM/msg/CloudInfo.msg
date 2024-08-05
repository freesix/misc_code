# Cloud Info
std_msgs/Header header 
# 在点云去畸变时把rang数据展开为一维向量
# ring代表第几条激光线数
int32[] start_ring_index # 记录第一个ring在一维向量中的起始索引
int32[] end_ring_index # 记录第一个ring在一维向量中的结束索引

int32[]  point_col_ind # point column index in range image(记录一个ring数据中每个点在原始range图片中属于哪一列)
float32[] point_range # point range  (所有点的range值，展开为一维数组)

int64 imu_available
int64 odom_available
# 当从imu原始数据中获取的roll、pitch、yaw角初始值
# 当前面imu_available为true，odom_available为false时，使用这个值初始化(作为这一帧点云的初值)
# Attitude for LOAM initialization
float32 imu_roll_init
float32 imu_pitch_init
float32 imu_yaw_init

# Initial guess from imu pre-integration
# 从imu_Preintegration，imu预积分模块中获取的与这一帧雷达最接近的出事位姿
# 当odom_available为true时，使用这个值初始化，优先级高于imu_xxx_init
float32 initial_guess_x
float32 initial_guess_y
float32 initial_guess_z
float32 initial_guess_roll
float32 initial_guess_pitch
float32 initial_guess_yaw

# Point cloud messages
# 去畸变后的点云，用于特征提取，在特征提取后被清空
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed
# featureExtraction模块提取的边缘特征点
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature
# featureExtraction模块提取的平面特征点
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature
