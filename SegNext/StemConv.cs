using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp;
using static System.Runtime.InteropServices.JavaScript.JSType;
using TorchSharp.Modules;

namespace SegNext
{
    public class StemConv : Module<Tensor, Tensor>
    {
        Sequential module;
        public StemConv(int in_channels, int out_channels, float bn_momentum = 0.99f) : base("")
        {
            this.module = nn.Sequential();
            this.module.append(nn.Conv2d(in_channels, out_channels / 2, kernelSize: (3, 3), stride: (2, 2), padding: (1, 1)));
            this.module.append(new NormLayer(out_channels / 2, Config.Norm_TypeEnum));
            this.module.append(nn.GELU());
            this.module.append(nn.Conv2d(out_channels / 2, out_channels, kernelSize: (3, 3), stride: (2, 2), padding: (1, 1)));
            this.module.append(new NormLayer(out_channels, Config.Norm_TypeEnum));
            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }

        public override Tensor forward(Tensor x)
        {
            x = module.forward(x);
            return x;
        }
    }
}
