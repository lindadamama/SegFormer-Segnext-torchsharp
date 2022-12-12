using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp;
using static Tensorboard.TensorShapeProto.Types;
using System.Text.RegularExpressions;

namespace SegNext
{
    public class DWConv3x3 : Module<Tensor, Tensor>
    {
        Module<Tensor, Tensor> DWConv;
        public DWConv3x3 (int dim= 768,  string name = "") : base(name)
        {
            this.DWConv = nn.Conv2d(dim, dim, 3, 1, 1, bias : true, groups : dim);

            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }

        public override Tensor forward(Tensor x)
        {
            return this.DWConv.forward(x);
        }
    }
}
