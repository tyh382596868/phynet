import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

class my_Sigmoid(torch.nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}


    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Sigmoid.png

    Examples::

        >>> m = nn.Sigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(input)*torch.pi    

class UnetGeneratorDouble(nn.Module):

    def __init__(self):
        super(UnetGeneratorDouble,self).__init__()
        self.model = nn.Sequential(
            UnetGenerator(),
            UnetGenerator()
        )

    def forward(self,x):
        return self.model(x)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=1, output_nc=1, num_downs=9, ngf=32, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        # self.model = unet_block

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
    
    
class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU()
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, my_Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class BasicUnet(nn.Module):

    def __init__(self):

        super(BasicUnet, self).__init__()

        self.layer_01 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                1, 32, (3, 3), stride = (1, 1), padding = 0
                ),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            
            torch.nn.Conv2d(32, 32, (3, 3), stride = (1, 1), padding = 0),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU()
            )
        
        self.layer_01_maxpool = torch.nn.MaxPool2d(
            kernel_size = (2, 2), stride = (2, 2)
            )
        
        self.layer_02 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, (3, 3), stride = (1, 1), padding = 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            
            torch.nn.Conv2d(64, 64, (3, 3), stride = (1, 1), padding = 2),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU()
            )
        
        self.layer_02_maxpool = torch.nn.MaxPool2d(
            kernel_size = (2, 2), stride = (2, 2)
            )
        
        self.layer_03 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, (3, 3), stride = (1, 1), padding = 0),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            
            torch.nn.Conv2d(128, 128, (3, 3), stride = (1, 1), padding = 2),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU()
            )
        
        self.layer_03_maxpool = torch.nn.MaxPool2d(
            kernel_size = (2, 2), stride = (2, 2)
            )
        
        self.layer_04 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, (3, 3), stride = (1, 1), padding = 0),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(),
            
            torch.nn.Conv2d(256, 256, (3, 3), stride = (1, 1), padding = 2),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU()
            )
        
        self.layer_04_maxpool = torch.nn.MaxPool2d(
            kernel_size = (2, 2), stride = (2, 2)
            )
        
        self.layer_05 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, (3, 3), stride = (1, 1), padding = 0),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(),
            
            torch.nn.Conv2d(512, 512, (3, 3), stride = (1, 1), padding = 2),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU()
            )
        
        self.layer_part1 = torch.nn.Sequential(
            self.layer_01, self.layer_01_maxpool, 
            self.layer_02, self.layer_02_maxpool, 
            self.layer_03, self.layer_03_maxpool, 
            self.layer_04, self.layer_04_maxpool, self.layer_05
            )

    def forward(self, x):
        x = self.layer_part1(x)
        # x = self.conv2(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # output = self.out(x)
        return x

from collections import OrderedDict

import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")


        self.layer_10 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1),
            torch.nn.BatchNorm2d(1),
            # torch.nn.LeakyReLU()
            my_Sigmoid()
            )

        # self.conv = nn.Conv2d(
        #     in_channels=features, out_channels=out_channels, kernel_size=1
        # )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.layer_10(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

if __name__ == '__main__':
    X = torch.randn(1,1,512,512).to('cuda')
    print(X.shape)

    # from unetgood import UnetGenerator
    net = UNet().cuda()
    # UnetGeneratorDouble().to('cuda')
    y = net(X)
    print(y.shape)

    print(net)

    # net = BasicUnet().to('cuda')

    # for layer in net.layer_01:
    #     X = layer(X)
    #     print(layer.__class__.__name__,'output shape: \t',X.shape)
    # x = net.layer_01(x)
    
    # print(f'net.layer_01:{x.shape}')

    # x = net.layer_01_maxpool(x)

    # print(f'net.layer_01_maxpool:{x.shape}')






