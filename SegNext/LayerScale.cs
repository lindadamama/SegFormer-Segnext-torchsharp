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
    public class LayerScale : Module<Tensor, Tensor>
    {
        Tensor layer_scale;
        float init_value;
        public LayerScale(int in_channels, float init_value = 1e-2f) : base("")
        {
            var tensor = init_value * torch.ones(in_channels);
            this.layer_scale = nn.Parameter(tensor, requires_grad: true);
            this.init_value = init_value;
            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }


        public override Tensor forward(Tensor x)
        {
            if (this.init_value == 0)
            {
                return x;
            }
            else 
            {
                var scale =this.layer_scale.unsqueeze(-1).unsqueeze(-1);
                return x*scale;    
            }
        }
    }
}
