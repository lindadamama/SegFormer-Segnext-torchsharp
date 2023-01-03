using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace SwinUnet
{
    public class AttentionModule : Module<Tensor, Tensor>
    {
        Sequential net;
        public AttentionModule(int in_channels, int out_channels) : base("")
        {

            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }

        public override Tensor forward(Tensor x)
        {

            return this.net.forward(x);
        }
    }
}