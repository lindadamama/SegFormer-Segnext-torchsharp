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
using System.Text.RegularExpressions;
using static Tensorboard.ApiDef.Types;

namespace SegNext
{
    public class MSCA : Module<Tensor, Tensor>
    {
        Sequential module;
        Module<Tensor, Tensor> conv0;
        Module<Tensor, Tensor> conv0_1;
        Module<Tensor, Tensor> conv0_2;
        Module<Tensor, Tensor> conv1_1;
        Module<Tensor, Tensor> conv1_2;
        Module<Tensor, Tensor> conv2_1;
        Module<Tensor, Tensor> conv2_2;
        Module<Tensor, Tensor> conv3;
        public MSCA(int dim) : base("")
        {
            module = nn.Sequential();
            conv0 = nn.Conv2d(dim, dim, 5, padding: 2, groups: dim);
            conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding: (0, 3), groups: dim);
            conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding: (3, 0), groups: dim);
            conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding: (0, 5), groups: dim);
            conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding: (5, 0), groups: dim);
            conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding: (0, 10), groups: dim);
            conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding: (10, 0), groups: dim);
            conv3 = nn.Conv2d(dim, dim, 1);
            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }

        public override Tensor forward(Tensor x)
        {
            var u = x.clone();
            var attn = this.conv0.forward(x);
            var attn_0 = this.conv0_1.forward(attn);
            attn_0 = this.conv0_2.forward(attn_0);
            var attn_1 = this.conv1_1.forward(attn);
            attn_1 = this.conv1_2.forward(attn_1);
            var attn_2 = this.conv2_1.forward(attn);
            attn_2 = this.conv2_2.forward(attn_2);
            attn = attn + attn_0 + attn_1 + attn_2;
            attn = this.conv3.forward(attn);
            return attn * u;
        }
    }
}
