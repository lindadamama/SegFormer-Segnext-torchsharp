using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;
using static TorchSharp.torch.optim.lr_scheduler.impl.CyclicLR;

namespace SegNext
{
    public class StochasticDepth : Module<Tensor, Tensor>
    {
        Sequential module;
        public StochasticDepth(float p= 0.5f,string mode="row" ,  ) : base("")
        {

            RegisterComponents();
            if (Config.DeviceType == DeviceType.CUDA) this.to(DeviceType.CUDA);
        }

        //private Tensor(Tensor x, float p, string mode, bool isTraining) 
        //{


        //}
        public override Tensor forward(Tensor x)
        {

            return this.module.forward(x);
        }
    }
}
