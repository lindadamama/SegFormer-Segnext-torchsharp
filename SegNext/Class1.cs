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

    public class AttentionModule : Module<Tensor, Tensor>
    {
        Sequential net;
        public AttentionModule(int in_channels, int out_channels) : base("")
        {
            RegisterComponents();
            if(Config.DeviceType==DeviceType.CUDA) this.to(DeviceType.CUDA);
        }
        public override Tensor forward(Tensor x)
        {
            return this.net.forward(x);
        }
    }

      
}
