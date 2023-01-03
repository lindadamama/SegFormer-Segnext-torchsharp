using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;
using System.Text.RegularExpressions;

namespace SegNext
{
    public class ConvRelu : Module<Tensor, Tensor>
    {
        Sequential module;
        public ConvRelu(int in_channels, int out_channels, int kernel = 1, bool bias = false) : base("")
        {
            module = nn.Sequential();
            module.append(nn.Conv2d(in_channels, out_channels, kernelSize: kernel, groups: in_channels, bias: bias));
            module.append(nn.ReLU(inplace: true));
            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }
        public override Tensor forward(Tensor x)
        {
            return this.module.forward(x);
        }
    }
}
