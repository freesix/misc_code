import torch 
import torchvision

model = torch.load("/home/freesix/misc_code/torch_cpp/tools/modelall_best.pth",map_location="cpu")
model = model
test_data = {
        'x1':torch.rand(1,1000,2)-0.5,
        'x2':torch.rand(1,1000,2)-0.5,
        'desc1': torch.rand(1,1000,128),
        'desc2': torch.rand(1,1000,128)
        }

trace_model = torch.jit.trace(model,test_data)

trace_model.save("model.jit")
