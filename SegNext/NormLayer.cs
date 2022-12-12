using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp;
using static TorchSharp.torch.optim;
using F = TorchSharp.torch.nn.functional;


namespace SegNext
{
    public class NormLayer : Module<Tensor, Tensor>
    {
        Module<Tensor, Tensor> norm;
        public NormLayer(int in_channels, Norm_typeEnum norm_TypeEnum,   string name = "") : base(name)
        {
            if (norm_TypeEnum== Norm_typeEnum.batch_norm) 
            {
                this.norm=nn.BatchNorm2d(in_channels, eps : 1e-5, momentum :Config.BN_MOM);
            }
            if (norm_TypeEnum == Norm_typeEnum.layer_norm)
            {
                this.norm = new MyLayerNorm(in_channels);
            }
            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }

        public override Tensor forward(Tensor input1)
        {

            return this.norm.forward(input1);
        }
    }
}
