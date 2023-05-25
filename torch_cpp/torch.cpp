#include <iostream>
#include <torch/torch.h>
#include <Eigen/Core>

using namespace std;
int main(){
    torch::Tensor a = torch::ones({2,2});
    cout << a << endl;
}