using System;
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
    public class MyLayerNorm : Module<Tensor, Tensor>
    {
        Module<Tensor, Tensor> norm;
        public MyLayerNorm(int in_channels) : base("")
        {
            this.norm = nn.LayerNorm(in_channels, eps : 1e-5);
            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }

        public override Tensor forward(Tensor x)
        {
            var a =x.size();
            var b = a[0];
            var c = a[1];
            var h = a[2];
            var w = a[3];
            x = x.flatten(2).transpose(1, 2);
            x = this.norm.forward(x);
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous();
            return x;
        }
    }
}
