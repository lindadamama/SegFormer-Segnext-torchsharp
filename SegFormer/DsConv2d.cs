using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;
using static System.Runtime.InteropServices.JavaScript.JSType;
using System.Threading;
using System.Text.RegularExpressions;

namespace SegFormer
{
    public class DsConv2d : Module<Tensor, Tensor>
    {
        Sequential net;
        public DsConv2d(int dim_in, int dim_out, int kernel_size, int padding, int stride = 1, bool bias = true, string name = "") : base(name)
        {
            net = nn.Sequential();
            net.append(nn.Conv2d(dim_in, dim_in, kernelSize: kernel_size, padding: padding, groups: dim_in, stride: stride, bias: bias));
            net.append(nn.Conv2d(dim_in, dim_out, kernelSize: 1, bias: bias));
            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }
        public override Tensor forward(Tensor x)
        {

            return this.net.forward(x);
        }
    }
}
