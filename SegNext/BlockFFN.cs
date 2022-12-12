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
    public class BlockFFN : Module<Tensor, Tensor>
    {
        Sequential module;
        public BlockFFN(int in_channels, int out_channels, int hid_channels, float ls_init_val = 1e-2f, float drop_path = 0.0f, string name = "") : base(name)
        {
            this.module = nn.Sequential();
            this.module.append(new NormLayer(in_channels, Config.Norm_TypeEnum));
            this.module.append(new FFN(in_channels, out_channels, hid_channels));
            this.module.append(new LayerScale(in_channels, ls_init_val));
            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }
        public override Tensor forward(Tensor x)
        {
            var skip = x.clone();
            this.module.forward(x);
            return skip + x;
        }
    }
}
