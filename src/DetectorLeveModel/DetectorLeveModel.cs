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

            int cropW = crop.Width;
            int cropH = crop.Height;
            if (cropW <= 0 || cropH <= 0) return (double.NegativeInfinity, 0, 0, 0, 0);

            // Slide windows directly over the original crop coordinates to avoid
            // unstable scaling math. For each scale (work) we take square windows
            // of size `work` in crop-space, resize each window to 20x20 and
            // evaluate the model.
            for (int si = 0; si < scales.Length; si++)
            {
                int work = scales[si]; int step = steps[si];
                // skip scales larger than the crop
                if (work > cropW || work > cropH) continue;

                for (int y0 = 0; y0 + work <= cropH; y0 += step)
                {
                    for (int x0 = 0; x0 + work <= cropW; x0 += step)
                    {
                        var win = ManipuladorDeImagem.cortar(crop, x0, y0, work, work);
                        var t = ManipuladorDeImagem.TransformarCropParaTensorGrayscale(win, 20, ctx);
                        var outt = this.Forward(t, ctx);
                        double score = outt[0];
                        if (score < detectCutoff) continue;

                        // skip degenerate detections that cover essentially the whole crop
                        if (work >= Math.Max(0.99 * cropW, cropW) && work >= Math.Max(0.99 * cropH, cropH)) continue;

                        detections.Add((score, x0, y0, work, work, work));
                    }
                }
            }

            if (detections.Count == 0) return (double.NegativeInfinity, 0, 0, 0, 0);

            // Compute IoU helper for detections
            double IoU((double score, int x, int y, int w, int h, int work) a, (double score, int x, int y, int w, int h, int work) b)
            {
                int xa = Math.Max(a.x, b.x);
                int ya = Math.Max(a.y, b.y);
                int xb = Math.Min(a.x + a.w, b.x + b.w);
                int yb = Math.Min(a.y + a.h, b.y + b.h);
                int interW = xb - xa; int interH = yb - ya;
                if (interW <= 0 || interH <= 0) return 0.0;
                double inter = interW * interH;
                double union = a.w * a.h + b.w * b.h - inter;
                return inter / Math.Max(1.0, union);
            }

            // Simplified selection: prefer highest score; if scores are very close,
            // prefer detections found at smaller work (lower resolution windows),
            // then prefer smaller area.
            var detArray = detections.ToArray();
            int idx = 0;
            double bestScore = detArray[0].score;
            int bestWork = detArray[0].work;
            int bestArea = detArray[0].w * detArray[0].h;
            for (int i = 1; i < detArray.Length; i++)
            {
                var d = detArray[i];
                if (d.score > bestScore + 1e-12)
                {
                    idx = i; bestScore = d.score; bestWork = d.work; bestArea = d.w * d.h;
                }
                else if (Math.Abs(d.score - bestScore) <= 1e-12)
                {
                    // tie-break: prefer smaller work (i.e., detected at smaller resolution)
                    if (d.work < bestWork)
                    {
                        idx = i; bestWork = d.work; bestArea = d.w * d.h;
                    }
                    else if (d.work == bestWork)
                    {
                        // final tie-break: smaller area
                        var area = d.w * d.h;
                        if (area < bestArea)
                        {
                            idx = i; bestArea = area;
                        }
                    }
                }
            }

            var sel = detArray[idx];
            return (sel.score, sel.x, sel.y, sel.w, sel.h);
        }

        // Return top-K detections per scale (work). Output tuple includes work so
        // caller can color by scale. Returned list is flat: all detections from
        // all scales, each annotated with its 'work' value.
        public System.Collections.Generic.List<(double score, int x, int y, int w, int h, int work)> DetectTopPerScale(BMP crop, ComputacaoContexto ctx, double detectCutoff = 0.5, int[] scales = null, int[] steps = null, int topK = 5)
        {
            if (crop == null) throw new ArgumentNullException(nameof(crop));
            if (scales == null) scales = new int[] { 32, 48, 64 };
            if (steps == null) steps = new int[] { 4, 6, 8 };
            var results = new System.Collections.Generic.List<(double score, int x, int y, int w, int h, int work)>();

            int cropW = crop.Width;
            int cropH = crop.Height;
            if (cropW <= 0 || cropH <= 0) return results;

            for (int si = 0; si < scales.Length; si++)
            {
                int work = scales[si]; int step = steps[si];
                if (work > cropW || work > cropH) continue;
                var per = new System.Collections.Generic.List<(double score, int x, int y, int w, int h)>();
                for (int y0 = 0; y0 + work <= cropH; y0 += step)
                {
                    for (int x0 = 0; x0 + work <= cropW; x0 += step)
                    {
                        var win = ManipuladorDeImagem.cortar(crop, x0, y0, work, work);
                        var t = ManipuladorDeImagem.TransformarCropParaTensorGrayscale(win, 20, ctx);
                        var outt = this.Forward(t, ctx);
                        double score = outt[0];
                        if (score < detectCutoff) continue;
                        // ignore degenerate whole-crop boxes
                        if (work >= Math.Max(0.99 * cropW, cropW) && work >= Math.Max(0.99 * cropH, cropH)) continue;
                        per.Add((score, x0, y0, work, work));
                    }
                }
                // sort per-scale and take topK
                var top = per.OrderByDescending(p => p.score).Take(topK);
                foreach (var p in top) results.Add((p.score, p.x, p.y, p.w, p.h, work));
            }
            return results;
        }

        
    }
}
