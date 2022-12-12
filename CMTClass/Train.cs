using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System;
using System.IO;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using static TorchSharp.torch.utils.data;
using  F= TorchSharp.torch.nn.functional;

namespace CMTClass
{
    public class Train
    {
        public event Action<string> OnShowMsg;
        private void showMsg(string str)
        {
            this.OnShowMsg?.Invoke(str);
        }
        public event Action<(double, double)> OnShowTrainData;
        private void showTrainData(double loss, double acc)
        {
            OnShowTrainData?.Invoke((loss, acc));
        }
        public bool IsFinish { get; set; }=false;   
        public void TrainBatch(List<(Tensor, Tensor)> train_data, List<(Tensor, Tensor)> testData, Module<Tensor, Tensor> model, int epochs, int _trainBatchSize, int _testBatchSize, int _logInterval)
        {
            model.to(DeviceType.CUDA);
            using (var optimizer = torch.optim.Adam(model.parameters(), 0.001))
            {
                for (var epoch = 1; epoch <= epochs; epoch++)
                {
                    trainloop(model, optimizer, torch.nn.NLLLoss().to(DeviceType.CUDA), train_data,testData, epoch, train_data.Count(), _logInterval);
                    this.showMsg($"训练次数{epoch}");
                }
            }
            model.Dispose();
        }
        private void trainloop(
          Module<Tensor, Tensor> model,
          torch.optim.Optimizer optimizer,
          Loss<torch.Tensor, torch.Tensor, torch.Tensor> loss,
          List<(Tensor, Tensor)> trainData,
           List<(Tensor, Tensor)> testData,
          int epoch,

          long size,
          int _logInterval)
        {
           
            model.train();
            int batchId = 1;
            long total = 0;
            long correct = 0;
            Console.WriteLine($"Epoch: {epoch}...");
            var count = testData.Count;

            var  crit = nn.BCELoss().cuda();
            using (var d = torch.NewDisposeScope())
            {
                foreach (var data in trainData)
                {
                    optimizer.zero_grad();
                    var prediction = model.forward(data.Item1).to(DeviceType.CUDA);
                    var output = F.Sigmoid(prediction).to(DeviceType.CUDA);
                    //Console.WriteLine(output);  
                    ////var los= crit.forward(data.Item2, output);
                    //output.backward();

                    optimizer.step();

                    if (batchId % _logInterval == 0)
                    {
                        Console.WriteLine($"\rTrain: epoch {epoch} [{batchId } / {size}] Loss: {output.ToSingle():F4}");
                    }

                    batchId++;


                    //var lsm = log_softmax(prediction, 1).to(DeviceType.CUDA);
                    //var output = loss.forward(lsm, target);
                    //output.backward();
                    optimizer.step();
                    //total += target.shape[0];
                    //var predicted = prediction.argmax(1);
                    //correct += predicted.eq(target).sum().to(DeviceType.CUDA).ToInt64();
                   
                   
                    batchId++;
                    d.DisposeEverything();
                    if (IsFinish) break;
                }

            }
            if (batchId % _logInterval == 0 || total == size)
            {
                model.Save("123.pth");
                this.test(model, torch.nn.NLLLoss(), testData, count);
               
                //this.showMsg($"\rTrain: epoch {epoch} [{total} / {size}] Loss: {output.ToSingle().ToString("0.000000")} | Accuracy: {((float)correct / total).ToString("0.000000")}");
                //this.showTrainData(output.ToSingle(), (float)correct / total);
            }
        }

        private void test(
            Module<torch.Tensor, torch.Tensor> model,
            Loss<torch.Tensor, torch.Tensor, torch.Tensor> loss,
           List<(Tensor, Tensor)> testData,
            long size)
        {
            model.eval();
            double testLoss = 0;
            long correct = 0;
            int batchCount = 0;
            using (var d = torch.NewDisposeScope())
            {
                foreach (var data in testData)
                {
                    var target = data.Item2;
                    //model.to(DeviceType.CUDA);
                    //var prediction = model.forward(data.Item1);
                    //var lsm = log_softmax(prediction, 1);
                    //var output = loss.forward(lsm, target);
                    //testLoss += output.ToSingle();
                    batchCount += 1;
                    //var predicted = prediction.argmax(1);
                    //correct += predicted.eq(target).sum().ToInt64();
                    d.DisposeEverything();
                    if (IsFinish) break;
                }
                Console.WriteLine("Test");
            }
            Console.WriteLine($"\rTest set: Average loss {(testLoss / batchCount).ToString("0.0000")} | Accuracy {((float)correct / size).ToString("0.0000")}");
        }
    }
}
