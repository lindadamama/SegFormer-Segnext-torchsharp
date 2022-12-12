﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp;
using TorchSharp.Modules;

namespace SegNext
{
    public class FFN : Module<Tensor, Tensor>
    {
        Sequential m;
        public FFN(int in_channels, int out_channels,int hid_channels,  string name = "") : base(name)
        {
            this.m = nn.Sequential
                (
                nn.Conv2d(in_channels, hid_channels, 1),
                new DWConv3x3(hid_channels),
                nn.GELU(),
                nn.Conv2d(hid_channels, out_channels, 1)
                );
            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }

        public override Tensor forward(Tensor x)
        {
            return this.m.forward(x);
        }
    }
}