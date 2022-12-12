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
    public class WindowAttention : Module<Tensor, Tensor>
    {
        Sequential net;
        double scale = 0;
        Tensor relative_position_bias_table;
        Module<Tensor, Tensor> qkv;
        Module<Tensor, Tensor> attn_drop;
        Module<Tensor, Tensor> proj;
        Module<Tensor, Tensor> proj_drop;
        Module<Tensor, Tensor> softmax;
        int num_heads;
        int[] window_size;
        Tensor relative_position_index;
        Tensor mask;
        public WindowAttention(int dim, int[] window_size, int num_heads, float qk_scale, Tensor mask, bool qkv_bias = true, float attn_drop = 0.0f, float proj_drop = 0.0f, string name = "") : base(name)
        {
            

           
            var head_dim = dim / num_heads;
            this.num_heads = num_heads;
            this.window_size = window_size;
            this.mask = mask;
            if (qk_scale == 0)
            {
                this.scale = Math.Pow(head_dim, -0.5);
            }
            else
            {
                this.scale = qk_scale;
            }
            this.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads));
            var coords_h = torch.arange(window_size[0]);
            var coords_w = torch.arange(window_size[1]);
            var coords = torch.stack(torch.meshgrid(new List<Tensor>() { coords_h, coords_w }));
            var coords_flatten = torch.flatten(coords, 1);
            var relative_coords = coords_flatten[-1, -1, null] - coords_flatten[-1, null, -1]; ;
            relative_coords = relative_coords.permute(1, 2, 0).contiguous();
            relative_coords[-1, -1, 0] += window_size[0] - 1;
            relative_coords[-1, -1, 1] += window_size[1] - 1;
            relative_coords[-1, -1, 0] *= 2 * window_size[1] - 1;
            relative_position_index = relative_coords.sum(-1);
            this.register_buffer("relative_position_index", relative_position_index);
            this.qkv = nn.Linear(dim, dim * 3, hasBias: qkv_bias);
            this.attn_drop = nn.Dropout(attn_drop);
            this.proj = nn.Linear(dim, dim);
            this.proj_drop = nn.Dropout(proj_drop);
            this.softmax = nn.Softmax(dim: -1);
            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }

        public override Tensor forward(Tensor x)
        {
            var a = x.size();
            var B_ = a[0];
            var N = a[1];
            var C = a[2];
            var _qkv = this.qkv.forward(x).reshape(B_, N, 3, this.num_heads, C / this.num_heads).permute(2, 0, 3, 1, 4);
            var q = _qkv[0];
            var k = _qkv[1];
            var v = _qkv[2];
            q = q * this.scale;
            var attn = q.matmul(k.transpose(-2, -1));
            var relative_position_bias = this.relative_position_bias_table[this.relative_position_index.view(-1)].view(
                 this.window_size[0] * window_size[1], this.window_size[0] * this.window_size[1], -1);
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous();
            attn = attn + relative_position_bias.unsqueeze(0);
            if (mask.size()[0] != 0)
            {
                var nW = mask.shape[0];
                attn = attn.view(B_ / nW, nW, this.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0);
                attn = attn.view(-1, this.num_heads, N, N);
                attn = this.softmax.forward(attn);
            }
            else
            {
                attn = this.softmax.forward(attn);
            }
            attn = this.attn_drop.forward(attn);
            x = attn.matmul(v).transpose(1, 2).reshape(B_, N, C);
            x = this.proj.forward(x);
            x = this.proj_drop.forward(x);
            return x;

        }
    }
}
