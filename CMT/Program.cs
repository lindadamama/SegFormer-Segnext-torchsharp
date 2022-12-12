using CMTClass;
using System.IO;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using OpenCvSharp;


Mat mat = Mat.Zeros(300,400,MatType.CV_8UC1);
var contours = new List<List<Point>>();
contours.Add(new List<Point>() { new Point() {X=100,Y=100 }, new Point() { X = 110, Y = 120 }, new Point() { X = 110, Y = 130 } });    
Cv2.DrawContours(mat,contours,-1,255);
Cv2.ImShow("m",mat);
Cv2.WaitKey();

var x = torch.randn(2, 1, 7, 3).to(DeviceType.CUDA);
var conv = torch.nn.Conv2d(1, 120, (2, 3)).to(DeviceType.CUDA);
var a = x[-1, -1, 0];
Console.WriteLine(a);
Console.WriteLine(conv.forward(x));
//Console.WriteLine(torch.__version__);
//Console.WriteLine(torch.cuda_is_available());
//var cmt = new CMT();
//var train = new Train();
//train.OnShowMsg += msg =>
//{
//    Console.WriteLine(msg);
//};
//train.OnShowTrainData += data =>
//{
//    Console.WriteLine("loss:" + data.Item1);
//    Console.WriteLine("Acc:" + data.Item2);
//};
//var trainData = new List<(Tensor, Tensor)>();
//var testData = new List<(Tensor, Tensor)>();
//for (int i = 0; i < 10; i++)
//{
//    var a = torch.rand(224 * 224 * 3).to(DeviceType.CUDA);
//    a = torch.reshape(a, new long[] { 1, 3, 224, 224 }).to(DeviceType.CUDA); ;
//    var b = torch.rand(224 * 224 * 1).to(DeviceType.CUDA);
//    b = torch.reshape(b, new long[] { 1, 1, 224, 224 }).to(DeviceType.CUDA); ;
//    trainData.Add((a, b));
//    testData.Add((a, b));
//}
//train.TrainBatch(trainData, testData, cmt, 800, 2, 2, 2);
//Console.WriteLine("Hello, World!");
Console.ReadLine(); 
