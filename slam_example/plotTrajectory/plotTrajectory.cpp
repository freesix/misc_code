#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <unistd.h>
#include <Eigen/Geometry>
#include <pangolin/gl/gl.h>

using namespace std;
using namespace Eigen;

//读取轨迹信息
string trajectory_file = "/home/freesix/misc_code/slam_example/plotTrajectory/trajectory.txt";

void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>); 

int main(int argc, char **argv){
    vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses; //定义了一个容器，存储类型为Isometry3d，使用了一个内存分配器

    ifstream fin(trajectory_file); //创建一个名为fin的ifstream类，并打开文件
    if (!fin){
        cout << "connot find trajectory file at " << trajectory_file << endl;
        return 1;
    
    }

    while (!fin.eof()){ //ifstream类中的eof()方法用于判断fin对象中文件是否已经到文件尾
        double time, tx, ty, tz, qx, qy, qz, qw; //定义文件中对应变量
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw; //按文件格式提取数据，>>这里表示从输入流读取
        Isometry3d Twr(Quaterniond(qw,qx,qy,qz)); //表示定义了一个Isometry3d对象，并定义一个四元数作为变量
        Twr.pretranslate(Vector3d(tx,ty,tz)); //对当前的等距变换进行平移操作
        poses.push_back(Twr); //表示在poses容器末尾添加元素
    }
    cout << "read toral " << poses.size() << "pose entries" << endl;

    // 画出轨迹图

    DrawTrajectory(poses);
    return 0;
}

void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses){
    // 创建pangolin窗口并画出轨迹图
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768); //创建窗口并命名
    glEnable(GL_DEPTH_TEST);//启用深度测试，控制在渲染过程中是否考虑物体的深度信息
    glEnable(GL_BLEND); //启用混合，将源颜色与目标颜色混合，从而产生半透明的效果
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); //控制颜色混合方式


    pangolin::OpenGlRenderState s_cam( //定义了一个名为s_cam的管理OpenGL渲染状态的对象
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000), //投影矩阵
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0) //模型视图矩阵,用于将场景中的三维点变换到相机坐标系下
    );

    pangolin::View &d_cam = pangolin::CreateDisplay().SetBounds(0.0,1.0,0.0,1.0,-1024.0f/768.0f).
    SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glLineWidth(2);
        for (size_t i=0; i<poses.size(); i ++){
            // 画出每个位姿的三个坐标轴
            Vector3d Ow = poses[i].translation();
            Vector3d Xw = poses[i] * (0.1 * Vector3d(1,0,0));
            Vector3d Yw = poses[i] * (0.1 * Vector3d(0,1,0));
            Vector3d Zw = poses[i] * (0.1 * Vector3d(0,0,1));
            glBegin(GL_LINES);
            glColor3f(1.0,0.0,0.0);
            glVertex3d(Ow[0],Ow[1],Ow[2]);
            glVertex3d(Xw[0],Xw[1],Xw[2]);
            glColor3f(0.0,1.0,0.0);
            glVertex3d(Ow[0],Ow[1],Ow[2]);
            glVertex3d(Yw[0],Yw[1],Yw[2]);
            glColor3f(0.0,0.0,1.0);
            glVertex3d(Ow[0],Ow[1],Ow[2]);
            glVertex3d(Zw[0],Zw[1],Zw[2]);
            glEnd();
            
        }
        // 画出连线
        for (size_t i=0; i<poses.size(); i++){
            glColor3f(0.0,0.0,0.0);
            glBegin(GL_LINES);
            auto p1 = poses[i], p2 = poses[i+1];
            glVertex3d(p1.translation()[0],p1.translation()[1],p1.translation()[2]);
            glVertex3d(p2.translation()[0],p2.translation()[1],p2.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);
        
    }
}