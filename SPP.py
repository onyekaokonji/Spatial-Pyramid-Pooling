import math
import torch
import torch.nn as nn

def SPP(self, n_samples, previous_conv_layer, previous_conv_layer_size, output_pool_size):
    for i in range(len(output_pool_size)):
        window_height = int(math.ceil(previous_conv_layer_size[0]/output_pool_size[i]))
        window_width = int(math.ceil(previous_conv_layer_size[1]/output_pool_size[i]))
        window_stride_x = int(math.floor(previous_conv_layer_size[0]/output_pool_size[i]))
        window_stride_y = int(math.floor(previous_conv_layer_size[1]/output_pool_size[i]))
        window_height_pad = int((window_height*output_pool_size[i] - previous_conv_layer_size[0] + 1)/2)
        window_width_pad = int((window_width*output_pool_size[i] - previous_conv_layer_size[1] + 1)/2)

        maxpool = nn.MaxPool2d(kernel_size=(window_height, window_width), stride=(window_stride_x, window_stride_y), padding = (window_height_pad, window_width_pad))
        x = maxpool(previous_conv_layer)
        if i == 0:
            spp = x.view(n_samples, -1)
            
        else:
            spp = torch.cat((spp, x.view(n_samples, -1)), 1)

    return spp

