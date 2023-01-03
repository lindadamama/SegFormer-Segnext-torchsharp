using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp;
using Google.Protobuf.WellKnownTypes;
using System.Text.RegularExpressions;
using TorchSharp.Modules;

namespace SegNext
{
    public class ConvBNRelu : Module<Tensor, Tensor>
    {
        Sequential module;
        public ConvBNRelu(int in_channels, int out_channels, int kernel = 3, int stride = 1, string padding = "same", int dilation = 1, int groups = 1) : base("")
        {
            int pd = 1;
            if (padding == "same")
            {
                if (kernel == 1)
                {
                    pd = 0;
                }
                if (kernel == 3)
                {
                    pd = 1;
                }
            }
            this.module = nn.Sequential();
            this.module.append(nn.Conv2d(in_channels, out_channels, kernelSize: kernel, padding: pd, stride: stride, dilation: dilation, groups: groups, bias: false));
            this.module.append(new NormLayer(out_channels, Config.Norm_TypeEnum));
            this.module.append(nn.ReLU(inplace: true));
            this.RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }

        public override Tensor forward(Tensor x)
        {
            return this.module.forward(x);
        }
    }
}
