#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

int main(int argc, char **argv){
    Mat img1=imread("/home/freesix/misc_code/example_cmake/8.jpg");
    Mat img2=imread("/home/freesix/misc_code/example_cmake/11.jpg");

    Ptr<Feature2D> sift=SIFT::create(100); //创建sift
    vector<KeyPoint> kp1,kp2; //定义特征点
    Mat des1,des2; //定义视觉描述子
    sift->detectAndCompute(img1,noArray(),kp1,des1);
    sift->detectAndCompute(img2,noArray(),kp2,des2);
    
    Mat des1_point1;
    // des1_point1=des1.at<uchar>(1);
    // cout << des1_point1 << endl;
    des1_point1 = des1.row(2);
    cout << des1_point1 << endl;
    //特征点配对
    Ptr<DescriptorMatcher> flann = FlannBasedMatcher::create();
    vector<vector<DMatch>> matches,good_matches; //定义匹配
    flann->knnMatch(des1,des2,matches,2);
    vector<Point2f> src_pts,dst_pts;
    for(size_t i=0; i<matches.size(); i++){
        if(matches[i][0].distance<0.6*matches[i][1].distance){
           src_pts.push_back(kp1[matches[i][0].queryIdx].pt);
           dst_pts.push_back(kp2[matches[i][0].trainIdx].pt);
           good_matches.push_back(matches[i]); 
        } 
    }
    circle(img1,src_pts[0],10,Scalar(0,0,255),5);
    circle(img2,dst_pts[0],10,Scalar(0,0,255),5);
    Mat display;
    hconcat(img1,img2,display);
    for (int i=0; i<dst_pts.size(); i++){
        if(i==0){
            line(display,src_pts[0],Point2f (dst_pts[i].x+img1.rows,dst_pts[i].y),Scalar(0,255,0),int(dst_pts.size()-i)/2+6);
    
        }
        else if (0 < i < dst_pts.size()/10)
        {
            line(display,src_pts[0],Point2f (dst_pts[i].x+img1.rows,dst_pts[i].y),Scalar(0,255,0),int((dst_pts.size()-i)/5)+1);

        }
        else{
        
            line(display,src_pts[0],Point2f (dst_pts[i].x+img1.rows,dst_pts[i].y),Scalar(0,255,0),1);//int((dst_pts.size()-i)/4)+1);
        }
    
    }
    imwrite("display1.jpg",display);
    
    // line(src_pts[0],dst_pts,)
    
    // imshow("img1",img1);
    // imshow("img2",img2);
    // waitKey(0);

    // Mat score;
    // for (int i=0; i<matches[0].size(); i++){
    //     score.push_back(matches[0][i].distance);
    // }
    // Mat score_top;
    // normalize(score, score_top);
    // score_top=score_top/1;
    // circle(img1,kp1[matches[0][0].queryIdx].pt,10,Scalar(0,0,255),5);
    // circle(img2,kp2[matches[0][0].trainIdx].pt,10,Scalar(0,0,255),5);
    // imshow("img1",img1);
    // imshow("img2",img2);
    // waitKey(0);
    
    
    // normalize()
    
    return 0;
}