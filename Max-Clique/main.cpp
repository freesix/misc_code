#include "extractFeature.hpp"

int main(int argc, char** argv){
    Mat img1 = imread("/home/freesix/misc_code/Max-Clique/images/11.jpg",IMREAD_GRAYSCALE);
    Mat img2 = imread("/home/freesix/misc_code/Max-Clique/images/8.jpg",IMREAD_GRAYSCALE);

    vector<KeyPoint>kp1,kp2;
    Mat feature1,feature2;
    ImageFeatureExtrac(img1, kp1, feature1);
    ImageFeatureExtrac(img2, kp2, feature2);
    vector<vector<int>> distance;
    vector<DMatch> matchs;
    calcInitMatch(feature1, feature2, matchs);
    cout<<kp1.at(0).pt.x<<endl;
    CorrGraph graph;
    buildCorrGraph(matchs, graph, kp1, kp2);
    cout<<graph.getVertexNum()<<endl;
    cout<<graph.getEdgeNum()<<endl;

    return 0;

}


