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
    public class DWCONV : Module<Tensor, Tensor>
    {
        public string Name { get; set; }
        Module<Tensor, Tensor> conv;
        public DWCONV(int in_channels,int out_channels,int stride =1,int kernelSize= 3,int padding= 1, DeviceType device = DeviceType.CUDA, string name = "") : base(name)
        {
           
            this.Name = name;
            conv = nn.Conv2d(in_channels, out_channels, kernelSize,stride, padding, groups: in_channels,bias:true);
            RegisterComponents();
            if (device != null && device == DeviceType.CUDA)
                this.to(device);
        }
        public override Tensor forward(Tensor input1)
        {
            return conv.forward(input1);
        }
    }
}
