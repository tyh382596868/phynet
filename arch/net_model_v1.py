import torch
import torchvision

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
        
class net_model_v1(torch.nn.Module):
    
    def __init__(self):
        
        super(net_model_v1, self).__init__()
        
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
        
        #-------------------------------------------------------

        # layer_06
        
        self.layer_06_01 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                512, 256, (3, 3), stride = (2, 2), padding = 1, 
                output_padding = 1
                ),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU()
            )
            
        self.layer_06_02 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, (3, 3), stride = (1, 1), padding = 0),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU()
            )
        
        self.layer_06_03 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, (3, 3), stride = (1, 1), padding = 2),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU()
            )
        
        # layer_07
        
        self.layer_07_01 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                256, 128, (3, 3), stride = (2, 2), padding = 1, 
                output_padding = 1
                ),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU()
            )
            
        self.layer_07_02 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, (3, 3), stride = (1, 1), padding = 0),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU()
            )
        
        self.layer_07_03 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, (3, 3), stride = (1, 1), padding = 2),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU()
            )
        
        # layer_08
        
        self.layer_08_01 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                128, 64, (3, 3), stride = (2, 2), padding = 1, 
                output_padding = 1
                ),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU()
            )
            
        self.layer_08_02 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, (3, 3), stride = (1, 1), padding = 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU()
            )
        
        self.layer_08_03 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, (3, 3), stride = (1, 1), padding = 2),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU()
            )
        
        # layer_09
        
        self.layer_09_01 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                64, 32, (3, 3), stride = (2, 2), padding = 1, 
                output_padding = 1
                ),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU()
            )
            
        self.layer_09_02 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, (3, 3), stride = (1, 1), padding = 0),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU()
            )
        
        self.layer_09_03 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, (3, 3), stride = (1, 1), padding = 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU()
            )
        
        # layer_10
        
        self.layer_10 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 1, (3, 3), stride = (1, 1), padding = 1),
            torch.nn.BatchNorm2d(1),
            # torch.nn.LeakyReLU()
            my_Sigmoid()
            )
        
    def forward(self, x):
        
        #-------------------------------------------------------
        enc1 = self.layer_01(x)
        enc2 = self.layer_02(self.layer_01_maxpool(enc1))
        enc3 = self.layer_03(self.layer_02_maxpool(enc2))
        enc4 = self.layer_04(self.layer_03_maxpool(enc3))

        bottleneck = self.layer_05(self.layer_04_maxpool(enc4))
        
        dec4 = self.layer_06_01(bottleneck)
        # x6_1 = self.layer_04(
        #     self.layer_03_maxpool(self.layer_03(self.layer_02_maxpool(
        #         self.layer_02(self.layer_01_maxpool(self.layer_01(x)))
        #         )))
        #     )
        dec4 = torch.cat((dec4, enc4), 1)
        dec4 = self.layer_06_03(self.layer_06_02(dec4))
        
        #-------------------------------------------------------
        dec3 = self.layer_07_01(dec4)
        # x7_1 = self.layer_03(self.layer_02_maxpool(
        #     self.layer_02(self.layer_01_maxpool(self.layer_01(x)))
        #     ))
        dec3 = torch.cat((dec3, enc3), 1)
        dec3 = self.layer_07_03(self.layer_07_02(dec3))
        
        #-------------------------------------------------------
        dec2 = self.layer_08_01(dec3)
        # x8_1 = self.layer_02(self.layer_01_maxpool(self.layer_01(x)))
        dec2 = torch.cat((dec2, enc2), 1)
        dec2 = self.layer_08_03(self.layer_08_02(dec2))
        
        #-------------------------------------------------------
        dec1 = self.layer_09_01(dec2)
        # x9_1 = self.layer_01(x)
        dec1 = torch.cat((dec1, enc1), 1)
        dec1 = self.layer_09_03(self.layer_09_02(dec1))
        
        #-------------------------------------------------------
        x10 = self.layer_10(dec1)
        
        return x10
        