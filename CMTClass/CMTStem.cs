using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Channels;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static Tensorboard.CostGraphDef.Types;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;


namespace CMTClass
{
    public class CMTStem : Module<Tensor, Tensor>
    {
        public string Name { get; set; }
        Module<Tensor, Tensor> conv1;
        Module<Tensor, Tensor> gelu1;
        Module<Tensor, Tensor> bn1;
        Module<Tensor, Tensor> conv2;
        Module<Tensor, Tensor> gelu2;
        Module<Tensor, Tensor> bn2;
        Module<Tensor, Tensor> conv3;
        Module<Tensor, Tensor> gelu3;
        Module<Tensor, Tensor> bn3;
        public CMTStem(int in_channels,int out_channels,DeviceType device=DeviceType.CUDA) : base("")
        {

            this.Name = name;
            this.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride: 2, padding: 1, bias: false);
            this.gelu1 = nn.GELU();
            this.bn1 = nn.BatchNorm2d(out_channels);
            this.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride: 1, padding: 1, bias: false);
            this.gelu2 = nn.GELU();
            this.bn2 = nn.BatchNorm2d(out_channels);
            this.conv3 = nn.Conv2d(out_channels, out_channels, 3, stride: 1, padding: 1, bias: false);
            this.gelu3 = nn.GELU();
            this.bn3 = nn.BatchNorm2d(out_channels);
            RegisterComponents();
            if (device != null && device == DeviceType.CUDA)
                this.to(device);
        }
        public override Tensor forward(Tensor x)
        {
            x = this.conv1.forward(x);
            x = this.gelu1.forward(x);
            x = this.bn1.forward(x);
            x = this.conv2.forward(x);
            x = this.gelu2.forward(x);
            x = this.bn2.forward(x);
            x = this.conv3.forward(x);
            x = this.gelu3.forward(x);
            var result = this.bn3.forward(x);
            return result;

        }
    }
}
