import torch
import torch
from torchvision.models import AlexNet
from torchviz import make_dot

from net_model_v1 import net_model_v1




class Prop(torch.nn.Module):
    
    def __init__(self, dx=2.2e-6, dy=2.2e-6, lam=532e-9, dist=0.0788):
        
        super(Prop,self).__init__()
        
        self.dx = dx
        self.dy = dy
        self.lam = lam
        self.dist = dist
        
    def forward(self,img_phase):
        H = torch.exp(1j * img_phase) 
        fft_H = torch.fft.ifftshift(torch.fft.fft2(H))

        (Ny,Nx) = H.shape[0],H.shape[1]
        
        qx = torch.range(1-Nx/2, Nx/2, 1)
        qy = torch.range(1-Ny/2, Ny/2, 1)
        y, x = torch.meshgrid(qy, qx)
        r=(2*torch.pi*x/(self.dx*Nx))**2+(2*torch.pi*y/(self.dy*Ny))**2

        k=2*torch.pi/self.lam

        kernel=torch.exp(-1*1j*torch.sqrt(k**2-r)*self.dist).to(H.device)

        fft_HH=fft_H[:,:]*kernel
        fft_HH=torch.fft.fftshift(fft_HH)

        Ud=torch.fft.ifft2(fft_HH)

        Id=Ud
        Id1=torch.angle(Ud)
        intensity = torch.abs(Id) * torch.abs(Id)
    
        # print(f'torch.max(intensity){torch.max(intensity)}')
        # print(f'intensity{intensity}')

        return intensity
def prop(img,  dx=2.2e-6, dy=2.2e-6, lam=532e-9, dist=0.0788,device='cuda:0'):
    '''
    1.注意传播距离，确保生成图像的传播距离与训练的传播距离一致，一般默认一致
    2.输入数据的维度应该为二维，如果是三维，就算值一样傅里叶变换后也不一样
    '''

    # print(torch.max(img))

    img_phase = img #*torch.pi

    # print(torch.max(img_phase))
    
    H = torch.exp(1j * img_phase) 
    fft_H = torch.fft.ifftshift(torch.fft.fft2(H))
    # H = torch.exp(1j * img) 
    # (Ny,Nx) = H.size()
    # fft_H = torch.fft.ifftshift(torch.fft.fft2(H)).to(device)
    # the axis in frequency space
    # qx = torch.linspace(0.25/xstart, 0.25/xend, nx) * nx
    # qy = torch.linspace(0.25/ystart, 0.25/yend, ny) * ny
    
    # qx = torch.range(1-Nx/2, Nx/2, 1).to(device)
    # qy = torch.range(1-Ny/2, Ny/2, 1).to(device)
    # # print(qx)
    # y, x = torch.meshgrid(qx, qy)
    # # print(f'mesh_qx{mesh_qx}')
    # # print(f'mesh_qy{mesh_qy}')
    (Ny,Nx) = H.shape[0],H.shape[1]
    
    qx = torch.range(1-Nx/2, Nx/2, 1).to(device)
    qy = torch.range(1-Ny/2, Ny/2, 1).to(device)
    y, x = torch.meshgrid(qy, qx)
    r=(2*torch.pi*x/(dx*Nx))**2+(2*torch.pi*y/(dy*Ny))**2

    k=2*torch.pi/lam

    kernel=torch.exp(-1*1j*torch.sqrt(k**2-r)*dist)

    fft_HH=fft_H[:,:]*kernel
    fft_HH=torch.fft.fftshift(fft_HH)

    Ud=torch.fft.ifft2(fft_HH)

    Id=Ud
    Id1=torch.angle(Ud)
    intensity = torch.abs(Id) * torch.abs(Id)
 
    # print(f'torch.max(intensity){torch.max(intensity)}')
    # print(f'intensity{intensity}')

    return intensity    
if __name__== '__main__':
    x = torch.randn(1,1,512,512).to('cuda')
    
    # prop = Prop().to('cuda')
    net = net_model_v1().to('cuda')
    
    y = net(x)
    
    x1 = prop(y[0, 0, :, :])
    
    loss_mse = torch.nn.MSELoss()
    print(y.shape)
    
    loss = loss_mse(x.float(),x1.float())
    # 构造图对象，3种方式
    g = make_dot(loss)
    # g = make_dot(y, params=dict(model.named_parameters()))
    # g = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))

    # 保存图像
    # g.view()  # 生成 Digraph.gv.pdf，并自动打开
    g.render(filename='graph', view=False)  # 保存为 graph.pdf，参数view表示是否打开pdf
    
    
    # import torch
    # from torchvision.models import AlexNet
    # from torchviz import make_dot

    # # 以AlexNet为例，前向传播
    # x = torch.rand(8, 3, 256, 512)
    # model = AlexNet()
    # y = model(x)

    # # 构造图对象，3种方式
    # g = make_dot(y)
    # # g = make_dot(y, params=dict(model.named_parameters()))
    # # g = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))

    # # 保存图像
    # # g.view()  # 生成 Digraph.gv.pdf，并自动打开
    # g.render(filename='graph', view=False)  # 保存为 graph.pdf，参数view表示是否打开pdf


# import torch
# from torchvision.models import AlexNet
# from torchviz import make_dot

# # 以AlexNet为例，前向传播
# x = torch.rand(8, 3, 256, 512)
# model = AlexNet()
# y = model(x)

# # 构造图对象，3种方式
# g = make_dot(y)
# # g = make_dot(y, params=dict(model.named_parameters()))
# # g = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))

# # 保存图像
# # g.view()  # 生成 Digraph.gv.pdf，并自动打开
# g.render(filename='graph', view=False)  # 保存为 graph.pdf，参数view表示是否打开pdf
