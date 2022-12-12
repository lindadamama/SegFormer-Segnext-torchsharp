using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static Tensorboard.CostGraphDef.Types;
using static Tensorboard.TensorShapeProto.Types;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace CMTClass
{
    public class LPU:Module<Tensor,Tensor>
    {
        Module<Tensor, Tensor> DWConv;
        public LPU(int in_channels,int  out_channels, DeviceType device=DeviceType.CUDA, string name="") : base(name) 
        {
            this.DWConv = new DWCONV( in_channels,out_channels);
            RegisterComponents();
            if (device != null && device == DeviceType.CUDA)
                this.to(device);
        }

        public override Tensor forward(Tensor input1)
        {
            var a = this.DWConv.forward(input1);
            return a+input1;
        }
    }
}
