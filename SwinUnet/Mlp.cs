using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;
using static Tensorboard.CostGraphDef.Types;

namespace SwinUnet
{
    public class Mlp : Module<Tensor, Tensor>
    {
        Sequential net = nn.Sequential();
        public Mlp(int in_features, int hidden_features, int out_features, Module<Tensor, Tensor> act_layer, float drop = 0, string name = "") : base(name)
        {
            if (out_features == 0) 
            {
                out_features = in_features;
            }
            if (hidden_features == 0) 
            {
                hidden_features = in_features;
            }
            if (act_layer == null)
            {
                act_layer = nn.GELU();
            }
            net.append(nn.Linear(in_features, hidden_features));
            net.append(act_layer);
            net.append(nn.Dropout(drop));
            net.append(nn.Linear(hidden_features, out_features));
            net.append(nn.Dropout(drop));
            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }
        public override Tensor forward(Tensor x)
        {

            return this.net.forward(x);
        }
    }
}
