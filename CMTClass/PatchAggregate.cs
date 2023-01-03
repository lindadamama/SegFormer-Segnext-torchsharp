using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;
using TorchSharp;

namespace CMTClass
{
    public class PatchAggregate : Module<Tensor, Tensor>
    {
        Module<Tensor, Tensor> patch_aggregation;
        public string Name;
        public PatchAggregate(int in_channel, int out_channel, DeviceType device = DeviceType.CUDA) : base("")
        {
            this.name = name;
            patch_aggregation = new Conv2x2(in_channel, out_channel, stride: 2);
            RegisterComponents();
            if (device != null && device == DeviceType.CUDA)
                this.to(device);
        }
        public override Tensor forward(Tensor input1)
        {
            input1 = patch_aggregation.forward(input1);
            var a = input1.size();
            var c = a[1];
            var h = a[2];
            var w = a[3];
            return F.layer_norm(input1, new long[] { c, h, w });
        }
    }
}
