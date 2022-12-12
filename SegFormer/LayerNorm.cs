using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;
using static Tensorboard.TensorShapeProto.Types;

namespace SegFormer
{
    public class LayerNorm : Module<Tensor, Tensor>
    {
        Sequential net;
        Tensor g;
        Tensor b;
        float eps;
        public LayerNorm(int dim,float eps = 1e-5f, string name = "") : base(name)
        {
            this.eps = eps; 
            this.g = Parameter(torch.ones(1, dim, 1, 1));
            this.b = nn.Parameter(torch.zeros(1, dim, 1, 1));
            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }

        public override Tensor forward(Tensor x)
        {
            var std = torch.var(x,  unbiased : false).sqrt();
            var mean = torch.mean(x);
            return (x - mean) / (std + this.eps) * this.g + this.b;
        }
    }
}
