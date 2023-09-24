#include<extractFeature.hpp>
#include<math.h>


/**
 * @brief 用SIFT算法提取图片特征点
 * @param img: 输入图片
 * @param kp: 图片的特征点
 * @param Feature: 图片的特征描述子
*/
void ImageFeatureExtrac(Mat img, vector<KeyPoint> &kp, Mat &Feature){
    if(img.empty()){
        cout<<"images is empty!"<<endl;
        return;
    }

    // 创建sift角点检测器
    Ptr<SIFT> sift = SIFT::create(2000);
    sift->detect(img, kp, noArray());
    // 计算描述子
    sift->compute(img, kp, Feature);
}

/**
 * @brief 用opencv计算特征点之间的初始匹配
 * @param feature1: 图片1的特征描述子nx128
 * @param feature2: 图片2的特征描述子mx128
 * @param matches: 存放匹配结果
*/
void calcInitMatch(Mat feature1, Mat feature2, vector<DMatch> &matches){
    if(feature1.empty() || feature2.empty()){
        cout<<"feature is empty!"<<endl;
        return;
    }

    // 匹配特征点
    // vector<DMatch> match; // 存放匹配结果
    vector<vector<DMatch>> knnMatch; // 存放knn匹配结果

    BFMatcher matcher(NORM_L2);
    matcher.knnMatch(feature1, feature2, knnMatch, 2);

    for(size_t r=0; r<knnMatch.size(); r++){
        if(knnMatch[r][0].distance < 0.7*knnMatch[r][1].distance){ // 最邻近距离小于0.7倍数次邻近距离
            matches.push_back(knnMatch[r][0]);
        }
    }
}

/**
 * @brief 根据matches配对点构造一个关系图
 * @param matches: 匹配结果
 * @param graph: 关系图
*/
void buildCorrGraph(vector<DMatch> matches, CorrGraph &graph, 
                    vector<KeyPoint> kp1, vector<KeyPoint> kp2){
    if(matches.empty()){
        cout<<"matches is empty!"<<endl;
        return;
    }

    // 构造关系图
    for(size_t i=0; i<matches.size(); i++){
        graph.srcpoints.push_back(matches[i].queryIdx);
        graph.dstpoints.push_back(matches[i].trainIdx);
    }

    // 计算两个坐标点之间的距离
    for(size_t i=0; i<graph.srcpoints.size(); i++){
        // vector<double> temp1;
        // vector<double> temp2;
        vector<double> temp;
        for(size_t j=0; j<graph.srcpoints.size(); j++){
            double distance1 = sqrt(pow(kp1.at(graph.srcpoints[i]).pt.x-
            kp1.at(graph.srcpoints[j]).pt.x, 2) + pow(kp1.at(graph.srcpoints[i]).pt.y
            -kp1.at(graph.srcpoints[j]).pt.y, 2));
            // temp1.push_back(distance1);
            double distance2 = sqrt(pow(kp2.at(graph.dstpoints[i]).pt.x-
            kp2.at(graph.dstpoints[j]).pt.x, 2) + pow(kp2.at(graph.dstpoints[i]).pt.y
            -kp2.at(graph.dstpoints[j]).pt.y, 2));
            // temp2.push_back(distance2);
            temp.push_back(exp(-abs(distance1-distance2)/10000));
            
        }
        graph.first_weight.push_back(temp); // 一阶权重矩阵 
    }
    // 计算矩阵的叉乘
    for(int i=0; i<graph.first_weight.size(); i++){ // 第i行
        vector<double> temp1;
        vector<double> temp = graph.first_weight[i]; 
        for(int j=0; j<graph.first_weight[i].size(); j++){ // 第j列
            double total = 0;
            for(int k=0; k<temp.size(); k++){ // 第k个元素
                total = temp[k] * graph.first_weight[k][j] + total;
            }
            double temp2 = total * graph.first_weight[i][j]>
                    310 ? total * graph.first_weight[i][j]:0;
            // temp1.push_back(temp2>63?temp2:0);
            temp1.push_back(temp2);
        }
        graph.second_weight.push_back(temp1);
    }
}



