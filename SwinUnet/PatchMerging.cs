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

    public class PatchMerging : Module<Tensor, Tensor>
    {
        Sequential net;
        Module<Tensor, Tensor> norm_layer, reduction;
        int[] input_resolution;
        int dim;
        public PatchMerging(int[] input_resolution, int dim, Module<Tensor, Tensor> norm_layer, string name = "") : base(name)
        {
            this.dim = dim;
            this.input_resolution = input_resolution;
            this.norm_layer = nn.LayerNorm(4 * dim);
            reduction = nn.Linear(4 * dim, 2 * dim, hasBias: false);
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
            x = x.view(B, H, W, C);
            var x0 = x[TensorIndex.Ellipsis, TensorIndex.Slice(0, 2), TensorIndex.Slice(0, 2), TensorIndex.Ellipsis];
            var x1 = x[TensorIndex.Ellipsis, TensorIndex.Slice(1, 2), TensorIndex.Slice(1, 2), TensorIndex.Ellipsis];
            var x2 = x[TensorIndex.Ellipsis, TensorIndex.Slice(0, 2), TensorIndex.Slice(1, 2), TensorIndex.Ellipsis];
            var x3 = x[TensorIndex.Ellipsis, TensorIndex.Slice(1, 2), TensorIndex.Slice(1, 2), TensorIndex.Ellipsis];
            x = torch.cat(new List<Tensor>() { x0, x1, x2, x3 }, -1) ;
            x = x.view(B, -1, 4 * C);
            x = this.norm_layer.forward(x);
            x = this.reduction.forward(x);
            return x;
        }
    }
}
