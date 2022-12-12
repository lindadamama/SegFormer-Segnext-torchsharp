using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp;
using TorchSharp.Modules;

namespace SegNext
{
    public class HamDecoder : Module<Tensor, Tensor>
    {
        Sequential module;
        public HamDecoder(int in_channels, int out_channels,  int[] enc_embed_dims , string name = "") : base(name)
        {
            if (enc_embed_dims.Length== 0) 
            {
                enc_embed_dims = new int[] { 32, 64, 460, 256 };
            }
            var ham_channels =Config.Ham_channels;
            var count = 0;
            for (int i = 1; i < enc_embed_dims.Length; i++)
            {
                count += enc_embed_dims[i];
            }
            this.module = nn.Sequential
                (
                new ConvRelu(count, ham_channels),
                //new HamBurger(ham_channels, config),
                new ConvRelu(ham_channels, out_channels)

                );

            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }

        public override Tensor forward(Tensor input1)
        {
            
            return input1;
        }
    }
}
