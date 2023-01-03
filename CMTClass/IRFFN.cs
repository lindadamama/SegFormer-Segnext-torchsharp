using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace CMTClass
{
    public class IRFFN : Module<Tensor, Tensor>
    {
        public string Name { get; set; }
        public int kernel_size;
        Sequential conv1;
        Sequential dwconv;
        Sequential conv2;
        public IRFFN(int in_channels = 46, float R = 3.6f, DeviceType device = DeviceType.CUDA) : base("")
        {
            this.name = name;
            var exp_channels = Convert.ToInt32(in_channels * R);
            this.conv1 = nn.Sequential();
            this.conv1.append(nn.Conv2d(in_channels, exp_channels, kernel_size = 1));
            this.conv1.append(nn.BatchNorm2d(exp_channels));
            this.conv1.append(nn.GELU());
            this.dwconv = nn.Sequential();
            this.dwconv.append(new DWCONV(exp_channels, exp_channels));
            this.dwconv.append(nn.BatchNorm2d(exp_channels));
            this.dwconv.append(nn.GELU());
            this.conv2 = nn.Sequential();
            this.conv2.append(nn.Conv2d(exp_channels, in_channels, 1));
            this.conv2.append(nn.BatchNorm2d(in_channels));
            RegisterComponents();
            if (device != null && device == DeviceType.CUDA)
                this.to(device);
        }
        public override Tensor forward(Tensor x)
        {
            return x + this.conv2.forward(this.dwconv.forward(this.conv1.forward(x)));
        }
    }
}
