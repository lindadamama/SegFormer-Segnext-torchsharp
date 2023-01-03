using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp;

namespace CMTClass
{
    public class Conv2x2 : Module<Tensor, Tensor>
    {
        public string Name { get; set; }
        Module<Tensor, Tensor> conv;
        public Conv2x2(int in_channels, int out_channels, int stride = 1, DeviceType device = DeviceType.CUDA) : base("")
        {
            this.name = name;
            this.conv = nn.Conv2d(in_channels, out_channels, 2, stride, padding: 0, bias: true);
            RegisterComponents();
            if (device != null && device == DeviceType.CUDA)
                this.to(device);
        }
        public override Tensor forward(Tensor x)
        {

            return this.conv.forward(x);
        }
    }
}

