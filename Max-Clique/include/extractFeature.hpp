#pragma once
#ifndef EXTRACTFEATURE_HPP
#define EXTRACTFEATURE_HPP

#include<opencv2/opencv.hpp>
#include<vector>
#include<opencv2/xfeatures2d.hpp>
#include<iostream>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

// 定义一个关系图结构类
class CorrGraph{
    public:
    // 兼容图的匹配对
    vector<int> srcpoints;
    vector<int> dstpoints;
    // 权重矩阵
    vector<vector<double>> first_weight;
    vector<vector<double>> second_weight;
    // 获取图的顶点数
    int getVertexNum(){
        return srcpoints.size();
    }
    // 获取二阶图的边数
    int getEdgeNum(){
        int num = 0;
        for(int i=0; i<second_weight.size(); i++){
            for(int j=0; j<second_weight[i].size(); j++){
                if(second_weight[i][j] != 0){
                    num++;
                }
            }
        }
        return num/2;
    }
    
};


void ImageFeatureExtrac(Mat img, vector<KeyPoint> &kp, Mat &Feature);
void calcInitMatch(Mat feature1, Mat feature2, vector<DMatch> &matches);
void buildCorrGraph(vector<DMatch> matches, CorrGraph &graph, vector<KeyPoint> kp1, 
                    vector<KeyPoint> kp2);

    
#endif