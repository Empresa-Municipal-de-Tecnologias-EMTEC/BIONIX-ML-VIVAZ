using System;
using System.IO;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.dados.serializacao;
using Bionix.ML.dados.imagem;
using Bionix.ML.dados.imagem.bmp;
using System.Collections.Generic;
using System.Linq;

namespace DetectorLeveModel
{
    public class DetectorLeve
    {
        // MLP parameters: input 400 -> hidden 64 -> output 1
        public Tensor W1 { get; private set; }
        public Tensor b1 { get; private set; }
        public Tensor W2 { get; private set; }
        public Tensor b2 { get; private set; }

        public DetectorLeve() { }

        public void InitializeWeights(ComputacaoContexto ctx)
        {
            var fabrica = new FabricaTensor(ctx ?? new ComputacaoCPUContexto());
            W1 = fabrica.Criar(400, 64);
            b1 = fabrica.Criar(1, 64);
            W2 = fabrica.Criar(64, 1);
            b2 = fabrica.Criar(1, 1);
            // mark parameters as requiring gradients so autograd accumulates grads
            W1.RequiresGrad = true;
            b1.RequiresGrad = true;
            W2.RequiresGrad = true;
            b2.RequiresGrad = true;
            var rnd = new Random(1234);
            for (int i = 0; i < W1.Size; i++) W1[i] = (rnd.NextDouble() - 0.5) * 0.01;
            for (int i = 0; i < b1.Size; i++) b1[i] = 0.0;
            for (int i = 0; i < W2.Size; i++) W2[i] = (rnd.NextDouble() - 0.5) * 0.01;
            for (int i = 0; i < b2.Size; i++) b2[i] = 0.0;
        }

        // Forward: input is expected shape [h,w,1] (20x20x1). Returns scalar prob tensor [1,1].
        public Tensor Forward(Tensor input, ComputacaoContexto ctx)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            var fabrica = new FabricaTensor(ctx ?? new ComputacaoCPUContexto());
            // flatten to [1,400]
            var x = fabrica.Criar(1, input.Size);
            for (int i = 0; i < input.Size; i++) x[i] = input[i];

            // hidden = sigmoid(x * W1 + b1)
            var hidden = x.MatMul(W1); // [1,64]
            // add bias via tensor Add to preserve autograd graph
            hidden = hidden.Add(b1);
            // apply sigmoid elementwise (use CPU helper implementation via creating tensor)
            hidden = SigmoidTensor(hidden, ctx);

            // out = sigmoid(hidden * W2 + b2)
            var outt = hidden.MatMul(W2); // [1,1]
            outt = outt.Add(b2);
            outt = SigmoidTensor(outt, ctx);
            return outt;
        }

        private Tensor SigmoidTensor(Tensor input)
        {
            return SigmoidTensor(input, null);
        }

        private Tensor SigmoidTensor(Tensor input, ComputacaoContexto ctx)
        {
            var fabrica = new FabricaTensor(ctx ?? new ComputacaoCPUContexto());
            if (input is Bionix.ML.nucleo.tensor.TensorCPU srcCpu)
            {
                var outT = fabrica.Criar(srcCpu.Shape);
                for (int i = 0; i < srcCpu.Size; i++) outT[i] = 1.0 / (1.0 + Math.Exp(-srcCpu[i]));
                outT.RequiresGrad = true;
                outT.GradFn = new Bionix.ML.grafo.CPU.SigmoidFunction(srcCpu, outT as Bionix.ML.nucleo.tensor.TensorCPU);
                return outT;
            }
            else if (input is Bionix.ML.nucleo.tensor.TensorCPUSIMD srcSimd)
            {
                var outT = fabrica.Criar(srcSimd.Shape);
                int n = srcSimd.Size;
                if (System.Numerics.Vector.IsHardwareAccelerated)
                {
                    int vecSize = System.Numerics.Vector<double>.Count;
                    int i = 0;
                    for (; i <= n - vecSize; i += vecSize)
                    {
                        for (int k = 0; k < vecSize; k++) outT[i + k] = 1.0 / (1.0 + Math.Exp(-srcSimd[i + k]));
                    }
                    for (; i < n; i++) outT[i] = 1.0 / (1.0 + Math.Exp(-srcSimd[i]));
                }
                else
                {
                    for (int i = 0; i < n; i++) outT[i] = 1.0 / (1.0 + Math.Exp(-srcSimd[i]));
                }
                outT.RequiresGrad = true;
                outT.GradFn = new Bionix.ML.grafo.CPUSIMD.SigmoidFunction(srcSimd, outT as Bionix.ML.nucleo.tensor.TensorCPUSIMD);
                return outT;
            }
            else throw new NotSupportedException("Unsupported tensor type for Sigmoid helper.");
        }

        public void SaveWeights(string dir)
        {
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            try { SerializadorTensor.SaveBinary(Path.Combine(dir, "w1.bin"), W1); } catch { }
            try { SerializadorTensor.SaveBinary(Path.Combine(dir, "b1.bin"), b1); } catch { }
            try { SerializadorTensor.SaveBinary(Path.Combine(dir, "w2.bin"), W2); } catch { }
            try { SerializadorTensor.SaveBinary(Path.Combine(dir, "b2.bin"), b2); } catch { }
        }

        public void LoadWeights(string dir)
        {
            try
            {
                var p = Path.Combine(dir, "w1.bin"); if (File.Exists(p) && W1 != null) { var t = SerializadorTensor.LoadBinary(p); if (t != null && t.Size == W1.Size) for (int i=0;i<W1.Size;i++) W1[i]=t[i]; }
                p = Path.Combine(dir, "b1.bin"); if (File.Exists(p) && b1 != null) { var t = SerializadorTensor.LoadBinary(p); if (t != null && t.Size == b1.Size) for (int i=0;i<b1.Size;i++) b1[i]=t[i]; }
                p = Path.Combine(dir, "w2.bin"); if (File.Exists(p) && W2 != null) { var t = SerializadorTensor.LoadBinary(p); if (t != null && t.Size == W2.Size) for (int i=0;i<W2.Size;i++) W2[i]=t[i]; }
                p = Path.Combine(dir, "b2.bin"); if (File.Exists(p) && b2 != null) { var t = SerializadorTensor.LoadBinary(p); if (t != null && t.Size == b2.Size) for (int i=0;i<b2.Size;i++) b2[i]=t[i]; }
            }
            catch { }
        }

        public System.Collections.Generic.IEnumerable<(string name, Tensor tensor)> GetNamedParameters()
        {
            yield return ("w1", W1);
            yield return ("b1", b1);
            yield return ("w2", W2);
            yield return ("b2", b2);
        }

        // Singleton helper to reuse model instance across calls
        private static DetectorLeve _instance;
        private static readonly object _instLock = new object();

        public static DetectorLeve GetInstance(ComputacaoContexto ctx = null, string pesosDir = null)
        {
            if (_instance != null) return _instance;
            lock (_instLock)
            {
                if (_instance == null)
                {
                    var m = new DetectorLeve();
                    m.InitializeWeights(ctx ?? new ComputacaoCPUContexto());
                    if (!string.IsNullOrEmpty(pesosDir))
                    {
                        try { m.LoadWeights(pesosDir); } catch { }
                    }
                    _instance = m;
                }
            }
            return _instance;
        }

        // Perform multi-scale sliding-window detection on an input crop image.
        // crop: BMP image to search (arbitrary size)
        // ctx: computation context for tensor creation
        // detectCutoff: score threshold [0..1] to consider a window a detection
        // scales: working window sizes in pixels (e.g., 32,48,64)
        // steps: step sizes corresponding to scales
        public (double score, int x, int y, int w, int h) DetectBest(BMP crop, ComputacaoContexto ctx, double detectCutoff = 0.7, int[] scales = null, int[] steps = null)
        {
            if (crop == null) throw new ArgumentNullException(nameof(crop));
            if (scales == null) scales = new int[] { 32, 48, 64 };
            if (steps == null) steps = new int[] { 4, 6, 8 };
            var detections = new List<(double score, int x, int y, int w, int h, int work)>();
            for (int si = 0; si < scales.Length; si++)
            {
                int work = scales[si]; int step = steps[si];
                int minSideCrop = Math.Min(crop.Width, crop.Height);
                if (minSideCrop <= 0) continue;
                double scaleFactor = (double)work / (double)minSideCrop;
                int newW = Math.Max(1, (int)Math.Round(crop.Width * scaleFactor));
                int newH = Math.Max(1, (int)Math.Round(crop.Height * scaleFactor));
                var resized = ManipuladorDeImagem.redimensionar(crop, newW, newH);
                for (int y2 = 0; y2 + work <= newH; y2 += step)
                {
                    for (int x2 = 0; x2 + work <= newW; x2 += step)
                    {
                        var win = ManipuladorDeImagem.cortar(resized, x2, y2, work, work);
                        var t = ManipuladorDeImagem.TransformarCropParaTensorGrayscale(win, 20, ctx);
                        var outt = this.Forward(t, ctx);
                        double score = outt[0];
                        if (score < detectCutoff) continue;
                        double cxRes = x2 + work / 2.0;
                        double cyRes = y2 + work / 2.0;
                        double origCx = cxRes / scaleFactor;
                        double origCy = cyRes / scaleFactor;
                        double boxSize = work / scaleFactor;
                        int bw = Math.Max(1, (int)Math.Round(boxSize));
                        int bh = bw;
                        int bx = (int)Math.Round(origCx - bw / 2.0);
                        int by = (int)Math.Round(origCy - bh / 2.0);
                        bx = Math.Max(0, Math.Min(crop.Width - bw, bx));
                        by = Math.Max(0, Math.Min(crop.Height - bh, by));
                        detections.Add((score, bx, by, bw, bh, work));
                    }
                }
            }
            if (detections.Count == 0) return (double.NegativeInfinity, 0, 0, 0, 0);
            var best = detections.OrderByDescending(d => d.score).ThenByDescending(d => d.work).First();
            return (best.score, best.x, best.y, best.w, best.h);
        }

        
    }
}
