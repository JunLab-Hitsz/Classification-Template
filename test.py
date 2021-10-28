# 测试每个模块
import models
import torch
import models.ops as ops

if __name__ == '__main__':
    input = torch.randn(128,3,32,32)
    m = models.DeepConvNet(num_class=10, in_channels=3, img_height=32, img_width=32, conv=ops.CONV2DCAPS)
    output, decoder = m(input)
    print(output.size())
    print(decoder.size())
    
    
