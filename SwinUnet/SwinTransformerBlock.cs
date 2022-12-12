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
using static Tensorboard.TensorShapeProto.Types;
using static Tensorboard.ApiDef.Types;

namespace SwinUnet
{
    public class SwinTransformerBlock : Module<Tensor, Tensor>
    {
        Sequential net;
        Module<Tensor, Tensor> act_layer = nn.GELU();
        Module<Tensor, Tensor> norm_layer;
        Module<Tensor, Tensor> norm1;
        Module<Tensor, Tensor> attn;
        Module<Tensor, Tensor> mlp;
        Identity drop_path;
        Tensor attn_mask;
        Module<Tensor, Tensor> norm2;
        int dim;
        int[] input_resolution;
        int num_heads;
        int window_size;
        int shift_size;
        float mlp_ratio;
        public SwinTransformerBlock(int dim, int[] input_resolution, int num_heads, float qk_scale, int window_size = 7, int shift_size = 0,
                float mlp_ratio = 4, bool qkvbias = true, float drop = 0, float attn_drop = 0, float drop_path = 0, string name = "") : base(name)
        {

            this.dim = dim;
            this.input_resolution = input_resolution;
            this.num_heads = num_heads;
            this.window_size = window_size;
            this.shift_size = shift_size;
            this.mlp_ratio = mlp_ratio;
            if (this.input_resolution.Min() <= this.window_size)
            {
                this.shift_size = 0;
                this.window_size = this.input_resolution.Min();
            }
            norm1 = nn.LayerNorm(dim);
            attn = new WindowAttention(dim, new int[] { window_size, window_size }, num_heads: num_heads, qk_scale: qk_scale, qkv_bias: qkvbias, mask: new int[] { }, attn_drop: attn_drop, proj_drop: drop);
           
            this. norm2 = nn.LayerNorm(dim);
            int mlp_hidden_dim = Convert.ToInt32(dim * mlp_ratio);
            this. mlp = new Mlp(in_features: dim, hidden_features: mlp_hidden_dim, out_features: 0, act_layer: act_layer, drop: drop);
            if (shift_size > 0)
            {
                //    var H = input_resolution[0];
                //    var W = input_resolution[1];
                //    var  img_mask = torch.zeros(new long[] { 1, H, W, 1 });
                //    var h_slices = (slice(0, -window_size),
                //                slice(-window_size, -shift_size),
                //                slice(-shift_size, None));
                //    var w_slices = (slice(0, -window_size),
                //                 slice(-window_size, -shift_size),
                //                 slice(-shift_size, None));
                //   var  cnt = 0;
                //    //foreach (var h in h_slices) 
                //    //{
                //    //    foreach (var w in w_slices) 
                //    //    {
                //    //        img_mask[-1, h, w, -1] = cnt;
                //    //        cnt += 1;
                //    //    }

                //    //}


                //mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
                //mask_windows = mask_windows.view(-1, window_size * window_size)
                //attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                //attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            }
            else
            {
                attn_mask = null;
            }
            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }
        private Tensor window_partition(Tensor x, int window_size)
        {
            var a = x.size();
            var B = a[0];
            var H = a[1];
            var W = a[2];
            var C = a[2];
            x = x.view(B, H / window_size, window_size, W / window_size, window_size, C);
            var windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C);
            return windows;
        }

        Tensor window_reverse(Tensor windows, int window_size, int H, int W)
        {
            var B = (windows.shape[0] / (H * W / window_size / window_size));
            var x = windows.view(B, H / window_size, W / window_size, window_size, window_size, -1);
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1);
            return x;
        }
        public override Tensor forward(Tensor x)
        {
            var H = input_resolution[0];
            var W = input_resolution[0];
            var a = x.size();
            var B = a[0];
            var L = a[1];
            var C = a[2];
            if (L != W * H)
            {
                Console.WriteLine("参数错误");
            }
            var shortcut = x;
            x = this.norm1.forward(x);
            x = x.view(B, H, W, C);
            Tensor shifted_x;
            if (this.shift_size > 0)
            {
                shifted_x = torch.roll(x, shifts: (-this.shift_size, -this.shift_size), dims: (1, 2));
            }
            else
            {
                shifted_x = x;
            }
            var x_windows = window_partition(shifted_x, this.window_size);
            x_windows = x_windows.view(-1, this.window_size * this.window_size, C);
            var attn_windows = this.attn.forward(x_windows);
            attn_windows = attn_windows.view(-1, this.window_size, this.window_size, C);
            shifted_x = window_reverse(attn_windows, this.window_size, H, W);
            if (this.shift_size > 0) 
            {
                x = torch.roll(shifted_x, shifts: (this.shift_size, this.shift_size), dims: (1, 2));
            } 
            else
            {
                x = shifted_x;
            }
            x = x.view(B, H * W, C);
            x = shortcut + nn.Identity().forward(x);
            x = x + nn.Identity().forward(this.mlp.forward(this.norm2.forward(x)));
            return x;
        }
    }
}

