using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using System.Drawing;
using TorchSharp.Modules;
using TorchSharp;

namespace CMTClass
{
    public class CMT : Module<Tensor, Tensor>
    {
        public string Name { get; set; }
        Module<Tensor, Tensor> classifier;
        Module<Tensor, Tensor> stem;
        Module<Tensor, Tensor> patch1;
        Module<Tensor, Tensor> patch2;
        Module<Tensor, Tensor> patch3;
        Module<Tensor, Tensor> patch4;
        Module<Tensor, Tensor> avg_pool;
        Sequential stage1, stage2, stage3, stage4, fc;
        public CMT( int in_channels = 3,int stem_channel = 32, float R = 3.6f, int img_size = 224,int num_class = 10, DeviceType device=DeviceType.CUDA, string name="") : base(name)
        {
            this.name = name;
            var size = new int[] { img_size / 4, img_size / 8, img_size / 16, img_size / 32 }; 
            int[] cmt_channel = new int[] { 46, 92, 184, 368 };
            int[] patch_channel = new int[] { 46, 92, 184, 368 };
            int[] block_layer = new int[] { 3, 3, 16, 3 };
            this.stem = new CMTStem( in_channels, stem_channel);
            this.patch1 = new PatchAggregate( stem_channel, patch_channel[0]);
            this.patch2 = new PatchAggregate( patch_channel[0], patch_channel[1]);
            this.patch3 = new PatchAggregate( patch_channel[1], patch_channel[2]);
            this.patch4 = new PatchAggregate( patch_channel[2], patch_channel[3]);
            stage1 = nn.Sequential();
            var cmt_layer1 = new CMTBlock(img_size: size[0],
                        stride: 8,
                        d_k: cmt_channel[0],
                        d_v: cmt_channel[0],
                        num_heads: 1,
                        R = R,
                        in_channels = patch_channel[0]);
            for (int i = 0; i < block_layer[0]; i++)
            {
                stage1.append(cmt_layer1);
            }
            
            stage2 = nn.Sequential();
            for (int i = 0; i < block_layer[1]; i++)
            {
                var cmt_layer = new CMTBlock( img_size: size[1],
                        stride: 4,
                        d_k: cmt_channel[1] / 2,
                        d_v: cmt_channel[1] / 2,
                        num_heads: 2,
                        R = R,
                        in_channels = patch_channel[1]
                );
                stage2.append(cmt_layer);
            }
           
            this.stage3 = nn.Sequential();

            for (int i = 0; i < block_layer[2]; i++)
            {
                var cmt_layer = new CMTBlock(img_size: size[2],
                        stride: 2,
                        d_k: cmt_channel[2] / 4,
                        d_v: cmt_channel[2] / 4,
                        num_heads: 4,
                        R = R,
                        in_channels = patch_channel[2]
                );
                stage3.append(cmt_layer);
            }
           
            stage4 = nn.Sequential();
            for (int i = 0; i < block_layer[3]; i++)
            {
                var cmt_layer = new CMTBlock( img_size: size[3],
                        stride: 1,
                        d_k: cmt_channel[3] / 8,
                        d_v: cmt_channel[3] / 8,
                        num_heads: 8,
                        R = R,
                        in_channels = patch_channel[3]
                );
                stage4.append(cmt_layer);
            }
           
            this.avg_pool = nn.AdaptiveAvgPool2d(1);
            this.fc = nn.Sequential(
                nn.Linear(cmt_channel[3], 1280),
                nn.ReLU(inplace: true)
            );
            this.classifier = nn.Linear(1280, num_class);
         
            RegisterComponents();
            if (device != null && device == DeviceType.CUDA)
                this.to(device);
        }
        public override Tensor forward(Tensor x)
        {
            x = this.stem.forward(x);
            x = this.patch1.forward(x);
            x = this.stage1.forward(x);
            x = this.patch2.forward(x);
            x = this.stage2.forward(x);
            x = this.patch3.forward(x);
            x = this.stage3.forward(x);
            x = this.patch4.forward(x);
            x = this.stage4.forward(x);
            x = this.avg_pool.forward(x);
            x = torch.flatten(x, 1);
            x = this.fc.forward(x);
            x = this.classifier.forward(x);
            return x; ;
        }
    }
}
