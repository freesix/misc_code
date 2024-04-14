#include "extractClique.hpp"
#include <iostream>
#include "extractFeature.hpp"

using namespace std;

/**
 * @brief 回溯法求解最大团
 * @param graph: 关系图
 * @param clique: 最大团
*/
void backtrack(CorrGraph graph, vector<CorrGraph> &clique){
    int vertexNum = graph.getVertexNum();
    int edgeNum = graph.getEdgeNum();
    int i=1;
    int j;
    if(i>vertexNum){


    }
    

} 
/**
 * @brief 用BK算法寻找最大团
 * @param graph: 原始关系图
 * @param clique: 找出的最大团
 * @param d: 寻找的当前开始节点
*/
void BronKerbosch(CorrGraph graph, vector<CorrGraph> &clique, int d,  vector<int> R,
                  vector<int> P, vector<int> X){
    // vector<int> R;
    // vector<int> p;
    // vector<int> X;
    int MaxCliqueNums; // 极大团数量
    int vertexNum = graph.getVertexNum(); // 节点数
    int edgeNum = graph.getEdgeNum(); // 边数量

    if(P.empty() && X.empty()){
        for(int i:R){
            clique[MaxCliqueNums].srcpoints.push_back(graph.srcpoints[i]);
            clique[MaxCliqueNums].dstpoints.push_back(graph.dstpoints[i]);
            for(int j:R){
                if(graph.second_weight[i][j] != 0){
                    clique[MaxCliqueNums].second_weight[i][j] = graph.second_weight[i][j];
                }
            }
        }
        MaxCliqueNums++;
    } 

    int u = P[0]; // P为d节点的邻接节点集合
    for(int i=0; i<P.size(); ++i){
        int v = P[i]; //选中一个邻接节点
        if(graph.second_weight[u][v] != 0){
            continue;
        }
        for(int j=0; j<R.size(); ++j){
            R[]
        }
    }


}