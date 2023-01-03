using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace SegNext
{
    public class HamBurger : Module<Tensor, Tensor>
    {
        Sequential net;
        public HamBurger(int in_channels, int out_channels) : base("")
        {
            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }
        public override Tensor forward(Tensor x)
        {
            return this.net.forward(x);
        }
    }
}
