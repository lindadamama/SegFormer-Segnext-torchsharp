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
    public class FFN : Module<Tensor, Tensor>
    {
        Sequential m;
        public FFN(int in_channels, int out_channels,int hid_channels,  ) : base("")
        {
            this.m = nn.Sequential ();
            this.m.append(nn.Conv2d(in_channels, hid_channels, 1));
            this.m.append(new DWConv3x3(hid_channels));
            this.m.append(nn.GELU());
            this.m.append(nn.Conv2d(hid_channels, out_channels, 1));
            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }

        public override Tensor forward(Tensor x)
        {
            return this.m.forward(x);
        }
    }
}
