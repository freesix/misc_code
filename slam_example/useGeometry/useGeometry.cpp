#include <iostream>
#include <cmath>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

int main(int argc, char **argv){
    //定义3D旋转矩阵
    Matrix3d rotation_matrix = Matrix3d::Identity();
    AngleAxisd rotation_vector(M_PI/4, Vector3d(0,0,1)) ;
    cout.precision(3);
    cout << "rotation matrix = \n" << rotation_vector.matrix()<< endl;
    // 阵也可以直接赋值
    rotation_matrix = rotation_vector.toRotationMatrix();
    // 用AngleAxis进行坐标变换
    Vector3d v(1,0,0);
    Vector3d v_rotated = rotation_vector * v;
    cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;
    // 用旋转矩阵
    v_rotated = rotation_matrix * v;
    cout << "(1,0,0) after rotation(by matrix) = " << v_rotated.transpose() << endl;
    // 欧拉角：可以将旋转矩阵直接转换为欧拉角
    Vector3d euler_angles = rotation_matrix.eulerAngles(2,1,0);
    cout << "yam pitch roll =" << euler_angles.transpose() << endl;

    // 欧式变换矩阵
    Isometry3d T = Isometry3d::Identity();
    T.rotate(rotation_vector); //按照rotation_vector进行旋转
    T.pretranslate(Vector3d(1,3,4)); //设置平移向量
    cout << "Transform matrix =\n" << T.matrix() << endl;

    // 用便更换矩阵进行坐标变换
    Vector3d v_transformed = T*v;
    cout << "v transformed = " << v_transformed.transpose() << endl;
    
    // 对于仿射变换和射影变换，使用Eigen::Affine3d和Eigen::Projectived3d就可

    //四元数
    Quaterniond q = Quaterniond(rotation_vector);
    cout << "将旋转向量转换为四元数表达方式=" << q.coeffs().transpose() << endl;

    // 也可以从旋转矩阵转换
    q=Quaterniond(rotation_matrix);
    cout << "从旋转矩阵转换过来的四元数表示=" << q.coeffs().transpose() << endl;
    // 使用四元数旋转一个向量，使用重载的乘法即可
    v_rotated = q * v; //数学上是qvq^-1
    cout << "向量经过四元数旋转后=" << v_rotated.transpose() << endl;
    // 用常规向量乘法表示则是
    cout << "四元数向量方式计算旋转=" << (q*Quaterniond(0,1,0,0) * q.inverse()).coeffs().transpose() << endl;

    return 0;
    

}

    
