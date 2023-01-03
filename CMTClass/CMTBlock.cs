using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace CMTClass
{
    public class CMTBlock : Module<Tensor, Tensor>
    {
        public string Name { get; set; }
        public int kernel_size;
        Module<Tensor, Tensor> lpu;
        Module<Tensor, Tensor> lmhsa;
        Module<Tensor, Tensor> irffn;
        public CMTBlock(int img_size, int stride, int d_k, int d_v, int num_heads, float R = 3.6f, int in_channels = 46, DeviceType device = DeviceType.CUDA) : base("")
        {
            this.name = name;
            this.lpu = new LPU(in_channels, in_channels);
            this.lmhsa = new LMHSA(img_size, in_channels, d_k, d_v, stride, num_heads, 0.0f);
            this.irffn = new IRFFN(in_channels, R);
            RegisterComponents();
            if (device != null && device == DeviceType.CUDA)
                this.to(device);
        }
        public override Tensor forward(Tensor x)
        {
            x = this.lpu.forward(x);
            x = this.lmhsa.forward(x);
            x = this.irffn.forward(x);
            return x;
        }
    }
}
