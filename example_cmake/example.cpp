#include "opencv2/opencv.hpp"

using namespace cv;

int main(){
    // Mat img1=imread("/home/freesix/misc_code/example_cmake/demo_1.jpg");
    // Mat img2=imread("/home/freesix/misc_code/example_cmake/demo_2.jpg");
    Mat img1=imread("/home/freesix/misc_code/example_cmake/8.jpg");
    Mat img2=imread("/home/freesix/misc_code/example_cmake/11.jpg");

    Mat img11,img22;
    // Mat mask1=imread("/home/freesix/misc_code/example_cmake/demo_1.jpg",0);
    // Mat mask2=imread("/home/freesix/misc_code/example_cmake/demo_2.jpg",0);
    int hh=std::max(img1.rows,img2.rows);
    int ww=std::max(img1.cols,img2.cols);
    copyMakeBorder(img1,img11,hh-img1.rows,0,ww-img1.cols,0,BORDER_CONSTANT,Scalar(0,0,0));
    copyMakeBorder(img2,img22,hh-img2.rows,0,0,ww-img2.cols,BORDER_CONSTANT,Scalar(0,0,0));

    // Mat back1(ww+1,hh+1,img1.type(),Scalar(0,0,0));
    // Mat back2(ww+1,hh+1,img2.type(),Scalar(0,0,0));
    // Rect roi1(ww-img1.cols,img1.rows,img1.cols,img1.rows);
    // Rect roi2(0,img2.rows,img2.cols,img2.rows);
    // Mat re=back1(roi1);
    // imshow("re",re);
    // waitKey(0);
    // img1.copyTo(back1(roi1));
    // img2.copyTo(back2(roi2));


    Ptr<Feature2D> sift=SIFT::create();//创建SIFT对象
    std::vector<KeyPoint> kp1,kp2;
    Mat des1,des2;
    sift->detectAndCompute(img1,noArray(),kp1,des1);
    sift->detectAndCompute(img2,noArray(),kp2,des2);

    //画出提取的特征点
    Mat img1_kp,img2_kp;
    Scalar red(0,0,255);
    Scalar green(0,255,0);
    drawKeypoints(img1,kp1,img1_kp,red,DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // imshow("img1_kp",img1_kp);
    imwrite("img1_kp.jpg",img1_kp);
    drawKeypoints(img2,kp2,img2_kp,red,DrawMatchesFlags::DEFAULT);
    imwrite("img2_kp.jpg",img2_kp);

    //创建FLANN匹配器

    Ptr<DescriptorMatcher> flann = FlannBasedMatcher::create();

    //匹配特征点
    std::vector<std::vector<DMatch>> matches;
    flann->knnMatch(des1, des2, matches, 2);
    
    //使用RANSAXC算法进行匹配
    std::vector<Point2f> src_pts, dst_pts;
    std::vector<DMatch> good_matches;
    for(size_t i=0; i<matches.size(); i++){
        if(matches[i][0].distance < 0.5*matches[i][1].distance){ //第一邻近匹配点距离小于第二邻近点距离的0.7倍
            src_pts.push_back(kp1[matches[i][0].queryIdx].pt);
            dst_pts.push_back(kp2[matches[i][0].trainIdx].pt);
            good_matches.push_back(matches[i][0]); 
        }
    }
    // //画出对应匹配点，没有经过内外点判别的
    // Mat img_matches;
    // drawMatches(img1,kp1,img2,kp2,good_matches,img_matches,
    //             green,red,std::vector<char>(),DrawMatchesFlags::DEFAULT);
    // imwrite("../Good Matches.jpg",img_matches);

    //计算单应性矩阵并画出match结果
    std::vector<DMatch> ransac_good;
    Mat mask;
    Mat M=findHomography(src_pts,dst_pts,RANSAC,0.005,mask);
    Mat ransac_matches;
    hconcat(img11,img22,ransac_matches);
    // imshow("back",ransac_matches);
    // waitKey(0);
    
    // drawMatches(img1,kp1,img2,kp2,good_matches,ransac_matches,green,red,mask);
    float num_ture=0;
    for(int i=0; i<mask.rows; i++){
        if(mask.at<int>(0,i)){
            line(ransac_matches,src_pts[i],Point(dst_pts[i].x+ww,dst_pts[i].y),green,1,LINE_AA); 
            num_ture+=1;   
        }
        else{
            line(ransac_matches,src_pts[i],Point(dst_pts[i].x+ww,dst_pts[i].y),red,1,LINE_AA);
        }
    }
    std::cout << "ransac 算法正确率:" << (num_ture/mask.rows) << std::endl;
    imwrite("ran_matches.jpg",ransac_matches);

    
    // ransac_good=good_matches.index(mask);
    
    //计算图像大小
    int h=img1.rows, w=img1.cols;
    std::vector<Point2f> pts{{0,0},{0,h-1},{w-1,h-1},{w-1,0}};
    std::vector<Point2f> dst;
    perspectiveTransform(pts, dst, M); //根据单应矩阵计算变换后图像的四个点坐标
    int min_x = static_cast<int>(min(min(dst[0].x, dst[1].x), min(dst[2].x, dst[3].x)) - 0.5);
    int min_y = static_cast<int>(min(min(dst[0].y, dst[1].y), min(dst[2].y, dst[3].y)) - 0.5);
    int max_x = static_cast<int>(max(max(dst[0].x, dst[1].x), max(dst[2].x, dst[3].x)) + 0.5);
    int max_y = static_cast<int>(max(max(dst[0].y, dst[1].y), max(dst[2].y, dst[3].y)) + 0.5);


    // int tx = -min_x, ty = -min_y;
    int H = max_y - min_y, W = max_x - min_x;

    //计算变换后的图像
    Mat result;
    warpPerspective(img1,result,M,result.size(),INTER_LINEAR,BORDER_TRANSPARENT);
    imwrite("result.jpg",result);
    // waitKey(0);
    // img2(result)=result;
    // imshow("resule",img2);
    // waitKey(0);
    // warpPerspective(img1, result, M, result.size(), INTER_LINEAR, BORDER_TRANSPARENT);
    // imshow("img2",img2(Rect(tx,ty,W,H)));
    // imshow("result",result);
    // waitKey(0);
    
    // img2(Rect(tx, ty, W, H)).copyTo(result(Rect(tx, ty, W, H)));
    //显示结果
    // namedWindow("Result", WINDOW_NORMAL);
    // imshow("Result", result);
    // waitKey(0);
    // destroyAllWindows();
    
    return 0;
    
}


