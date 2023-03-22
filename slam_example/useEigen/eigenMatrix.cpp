#include <iostream>

using namespace std;

#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;


#define MATRIX_SIZE 50
    

int main(int argc, char **argv){
    // 声明一个2*3的float矩阵
    Matrix<float,2,3> matrix_23;
    // Eigen通过typedef提供了许多内置类型，不过底层仍然调用的Eigen::Matrix
    Vector3d v_3d;
    Matrix<float,3,1> vd_3d;
    Matrix3d matrix_33 = Matrix3d::Zero(); //初始化为零
    // 如果不确定矩阵大小，可以使用动态大小的矩阵
    Matrix<double,Dynamic,Dynamic> matrix_dynamic;
    // 更简单的是
    MatrixXd matrix_x;

    // 数据初始化
    matrix_23 << 1,2,3,4,5,6;
    cout << "matrix 2x3 form 1 to 6 : \n" << matrix_23 << endl;

    // 访问矩阵中的元素
    cout << "print matrix 2x3:" << endl;

    for (int i=0; i<2; i++){
        for (int j=0; j<3; j++){
            cout << matrix_23(i,j) << "\t";
            cout << endl;
        }
    }

    // 矩阵和向量相乘
    v_3d << 3,2,1;
    vd_3d << 4,5,6;

    // 但是在Eigen里面不能混合两种不同类型的矩阵,应该做显式转换后再运算
    Matrix<double,2,1> result = matrix_23.cast<double>() * v_3d;//这里做两个矩阵乘法，需要将matrix_23转换为double类型
    cout << "[1,2,3;4,5,6] * [3,2,1]=" << result.transpose() << endl;

    Matrix<float,2,1> result2 = matrix_23*vd_3d;
    cout << "[1,2,3;4,5,6]*[4,5,6]=" << result2.transpose() << endl;

    //随机矩阵
    matrix_33 = Matrix3d::Random();
    cout << "random matrix: \n" << matrix_33 << endl;
    cout << "sum:" << matrix_33.sum() << endl;
    cout << "迹：" << matrix_33.trace() << endl;
    cout << "数乘：" << 10*matrix_33 << endl;
    cout << "逆：" << matrix_33.inverse() << endl;
    cout << "行列式：" << matrix_33.determinant() << endl;

    // 特征值
    // 实对称矩阵可以保证对角化成功
    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
    cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
    cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;
    
    // 解方程
    // 求解matrix_NN*X=v_Nd方程，其中N为原先定义
    Matrix<double,MATRIX_SIZE,MATRIX_SIZE> matrix_NN = MatrixXd::Random(MATRIX_SIZE,MATRIX_SIZE);
    matrix_NN = matrix_NN * matrix_NN.transpose(); //保证半正定
    Matrix<double,MATRIX_SIZE,1> v_Nd = MatrixXd::Random(MATRIX_SIZE,1);

    clock_t time_start = clock(); //记时

    // 直接求逆
    Matrix<double,MATRIX_SIZE,1> x=matrix_NN.inverse() * v_Nd;
    cout << "直接求逆的时间：" << 1000*(clock()-time_start) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x=" << x.transpose() << endl;

    //用矩阵分解的方式来求解，速度快很多
    //QR分解
    time_start = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "QR分解求解时间：" << 1000*(clock()-time_start) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x=" << x.transpose() << endl;

    //对于正定矩阵，可以用cholesky分解
    time_start = clock();
    x = matrix_NN.ldlt().solve(v_Nd);
    cout << "cholesky分解求解时间：" << 1000*(clock()-time_start) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x=" << x.transpose() << endl;

    return 0;
    
    
    
    
}











