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

namespace SwinUnet
{
    public class PatchExpand : Module<Tensor, Tensor>
    {
        Sequential net;
        int[] input_resolution;
        int dim;
        Module<Tensor, Tensor> expand;
        Module<Tensor, Tensor> norm;
        public PatchExpand(int[] input_resolution, int dim, int dim_scale = 2) : base("")
        {
            this.input_resolution = input_resolution;
            this.dim = dim;
            if (dim_scale == 2)
            {
                this.expand = nn.Linear(dim, 2 * dim, hasBias: false);
            }
            else
            {
                this.expand = nn.Identity();
            }
            this.norm = nn.LayerNorm(dim / dim_scale);
            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }

        public override Tensor forward(Tensor x)
        {
            var H = input_resolution[0];
            var W = input_resolution[0];
            var a = x.size();
            var B = a[0];
            var L = a[1];
            var C = a[2];
            x = this.expand.forward(x);
            x = x.view(B, H, W, C);
            x = x.reshape(B,H*2,W*2,C/4);
            //x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1 : 2, p2 : 2, c : C / 4);
            x = x.view(B, -1, C / 4);
            x = this.norm.forward(x);
            return x;
        }
    }
}
