# CNN_2d:The is modify vgg16net for input image64x64x3 example based on tensorflow1.0
# MutilGPU:My Machine has two GTX1080.Train CNN on multiple GPUs,not only compute forward pass,but also compute backward pass,so use make_parallel function make data into two GPUs and get two GPUs results,than when minimize loss function set colocate_gtadients_with_ops flag to true.
