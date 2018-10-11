# MutiltGPU_CNN
> This is an examples of how to use mutilGPU training

The is modify vgg16net for input image64x64x3 example based on tensorflow1.0

![](mutilgpu.png)
## How to Use
My Machine has two GTX1080.Train CNN on multiple GPUs,not only compute forward pass,but also compute backward pass,so use make_parallel function make data into two GPUs and get two GPUs results,then when minimize loss function set colocate_gtadients_with_ops flag to true.calculate two gpu lost value seperately,and merge them on cpu,minimize the loss function and update weight.

## Contact
* https://github.com/junqiangchen
* email: 1207173174@qq.com
* WeChat Public number: 最新医学影像技术
