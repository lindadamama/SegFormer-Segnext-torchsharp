using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp;

namespace SegNext
{
    public class DownSample : Module<Tensor, Tensor>
    {
        Module<Tensor, Tensor> proj;
        public DownSample(int kernel_Size = 3,int stride= 2,int in_channels= 3,int embed_dim= 768) : base("")
        {
            this.proj = nn.Conv2d(in_channels, embed_dim, kernelSize: (kernel_Size, kernel_Size), stride :(stride,stride), padding : (kernel_Size / 2, kernel_Size / 2));
            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }
        public override Tensor forward(Tensor x)
        {
            return this.proj.forward(x);
        }
    }
}
