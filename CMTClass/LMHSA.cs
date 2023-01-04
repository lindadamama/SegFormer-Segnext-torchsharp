using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static Tensorboard.CostGraphDef.Types;
using static Tensorboard.TensorShapeProto.Types;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace CMTClass
{
    public class LMHSA : Module<Tensor, Tensor>
    {
        public string Name { get; set; }
        Module<Tensor, Tensor> dwconv_k;
        Module<Tensor, Tensor> dwconv_v;
        Module<Tensor, Tensor> fc_q;
        Module<Tensor, Tensor> fc_k;
        Module<Tensor, Tensor> fc_v;
        Module<Tensor, Tensor> fc_o;
        Parameter B;
        float scaled_factor;
        float num_patches;
        int heads;
        int d_k, d_v, channels;
        public LMHSA( int input_size, int channels, int d_k, int d_v, int stride, int heads, float dropout, DeviceType device=DeviceType.CUDA) : base("")
        {
            this.Name = name;
            this.heads = heads;
            this.channels = channels;
            this.d_k = d_k;
            this.d_v = d_v;
            this.dwconv_k = new DWCONV( channels, channels, stride : stride);
            this.dwconv_v = new DWCONV( channels, channels, stride : stride);
            this.fc_q = nn.Linear(channels, heads * d_k);
            this.fc_k = nn.Linear(channels, heads * d_k);
            this.fc_v = nn.Linear(channels, heads * d_v);
            this.fc_o = nn.Linear(heads * d_k, channels);
            this.scaled_factor = (float)torch.pow(d_k, -0.5f);
            this.num_patches = (float)torch.pow(d_k / stride, 2);
            var input_size1 = (float)input_size;
            var a = new long[] { 1, (long)heads, (long)Math.Pow(input_size1, 2), (long)Math.Pow(input_size / stride, 2) };
            this.B = nn.Parameter(rand(a), requires_grad: true);
            RegisterComponents();
            if (device != null && device == DeviceType.CUDA)
                this.to(device);
        }

        public override Tensor forward(Tensor x)
        {
            var b = x.shape[0];
            var c = x.shape[1];
            var h = x.shape[2];
            var w = x.shape[3];
            var x_reshape = x.view(b, c, h * w).permute(0, 2, 1);
            x_reshape = nn.functional.layer_norm(x_reshape, new long[] { b, h * w, c });
            var q = this.fc_q.forward(x_reshape);
            q = q.view(b, h * w, this.heads, this.d_k).permute(0, 2, 1, 3).contiguous();
            var k = this.dwconv_k.forward(x);
            var k_b = k.shape[0];
            var k_c = k.shape[1];
            var k_h = k.shape[2];
            var k_w = k.shape[3];
            k = k.view(k_b, k_c, k_h * k_w).permute(0, 2, 1).contiguous();
            k = this.fc_k.forward(k);
            k = k.view(k_b, k_h * k_w, this.heads, this.d_k).permute(0, 2, 1, 3).contiguous();
            var v = this.dwconv_v.forward(x);
            var v_b = v.shape[0];
            var v_c = v.shape[1];
            var v_h = v.shape[2];
            var v_w = v.shape[3];
            v = v.view(v_b, v_c, v_h * v_w).permute(0, 2, 1).contiguous();
            v = this.fc_v.forward(v);
            v = v.view(v_b, v_h * v_w, this.heads, this.d_v).permute(0, 2, 1, 3).contiguous();
            var attn = torch.einsum("... i d, ... j d -> ... i j", q, k) * this.scaled_factor;
            attn = attn + this.B;
            attn = F.softmax(attn,dim: -1);
            var result = torch.matmul(attn, v).permute(0, 2, 1, 3);
            result = result.contiguous().view(b, h * w, this.heads * this.d_v);
            result = this.fc_o.forward(result).view(b, this.channels, h, w);
            result = result + x;
            return result;
        }
    }
}
