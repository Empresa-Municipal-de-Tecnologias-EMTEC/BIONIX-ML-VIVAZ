using System;
using System.IO;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.dados.serializacao;
using Bionix.ML.dados.imagem;
using Bionix.ML.dados.imagem.bmp;
using System.Collections.Generic;
using System.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace DetectorLeveBModel
{
    public class DetectorLeve
    {
        // Computation context used to create the model's tensors
        private ComputacaoContexto _ctxUsed;

        // Convolution + MLP parameters:
        // conv: 3x3 kernel -> 16 channels (valid conv) (not trained here)
        // pooling -> 9x9x16 = 1296 flattened input to MLP
        public Tensor convW { get; private set; }
        public Tensor convB { get; private set; }

        // MLP parameters: input 1296 -> hidden 64 -> output 1
        public Tensor W1 { get; private set; }
        public Tensor b1 { get; private set; }
        public Tensor W2 { get; private set; }
        public Tensor b2 { get; private set; }

        public DetectorLeve() { }

        public void InitializeWeights(ComputacaoContexto ctx)
        {
            var actualCtx = ctx ?? new ComputacaoCPUContexto();
            _ctxUsed = actualCtx;
            var fabrica = new FabricaTensor(actualCtx);
            // conv weights: im2col representation: patchSize=3*3*1=9 -> [9,16]
            convW = fabrica.Criar(9, 16);
            convB = fabrica.Criar(1, 16);

            W1 = fabrica.Criar(1296, 64);
            b1 = fabrica.Criar(1, 64);
            W2 = fabrica.Criar(64, 1);
            b2 = fabrica.Criar(1, 1);
            // mark parameters as requiring gradients so autograd accumulates grads
            // conv weights enabled for training
            convW.RequiresGrad = true;
            convB.RequiresGrad = true;

            W1.RequiresGrad = true;
            b1.RequiresGrad = true;
            W2.RequiresGrad = true;
            b2.RequiresGrad = true;
            var rnd = new Random(1234);
            for (int i = 0; i < convW.Size; i++) convW[i] = (rnd.NextDouble() - 0.5) * 0.05;
            for (int i = 0; i < convB.Size; i++) convB[i] = 0.0;

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
            // input expected shape [h,w,1]
            if (input.Shape.Length != 3) throw new ArgumentException("Expected input shape [h,w,1]");
            int h = input.Shape[0];
            int w = input.Shape[1];
            int c = input.Shape[2];
            // convolution params
            int kh = 3, kw = 3;
            int outH = h - kh + 1;
            int outW = w - kw + 1;
            int outC = 16;
            int patchSize = kh * kw * c; // expected 9

            // im2col: build matrix [outH*outW, patchSize]
            var Xcol = fabrica.Criar(outH * outW, patchSize);
            for (int oy = 0; oy < outH; oy++)
            {
                for (int ox = 0; ox < outW; ox++)
                {
                    int row = oy * outW + ox;
                    int p = 0;
                    for (int ky = 0; ky < kh; ky++)
                    {
                        for (int kx = 0; kx < kw; kx++)
                        {
                            for (int ic = 0; ic < c; ic++)
                            {
                                int inIdx = ((oy + ky) * w + (ox + kx)) * c + ic;
                                Xcol[row * patchSize + p] = input[inIdx];
                                p++;
                            }
                        }
                    }
                }
            }

            // conv: outCol = Xcol * convW  => [outH*outW, outC]
            var outCol = Xcol.MatMul(convW);
            // add bias by tiling convB using ones vector
            var ones = fabrica.Criar(outH * outW, 1);
            for (int i = 0; i < ones.Size; i++) ones[i] = 1.0;
            var biasMat = ones.MatMul(convB); // [outH*outW, outC]
            outCol = outCol.Add(biasMat);
            // activation: use ReLU after conv to avoid early saturation
            var outAct = ReLUTensor(outCol, ctx);

            // max pooling 2x2 -> reduces outH,outW (18,18) to (9,9)
            int pH = 2, pW = 2;
            int pooledH = outH / 2;
            int pooledW = outW / 2;
            int pooledN = pooledH * pooledW; // 81
            // compute max over 2x2 windows for each channel
            var pooled = fabrica.Criar(pooledN, outC);
            for (int py = 0; py < pooledH; py++)
            {
                for (int px = 0; px < pooledW; px++)
                {
                    int prow = py * pooledW + px;
                    for (int oc = 0; oc < outC; oc++)
                    {
                        double mval = double.NegativeInfinity;
                        for (int dy = 0; dy < pH; dy++)
                        {
                            for (int dx = 0; dx < pW; dx++)
                            {
                                int oy = py * pH + dy;
                                int ox = px * pW + dx;
                                int src = oy * outW + ox;
                                int idx = src * outC + oc;
                                var v = outAct[idx];
                                if (v > mval) mval = v;
                            }
                        }
                        pooled[prow * outC + oc] = mval;
                    }
                }
            }

            // flatten pooled [pooledN, outC] -> x [1, pooledN*outC]
            int flatDim = pooledN * outC; // 1296
            var x = fabrica.Criar(1, flatDim);
            for (int r = 0; r < pooledN; r++)
            {
                for (int oc = 0; oc < outC; oc++)
                {
                    int srcIdx = r * outC + oc;
                    int dstIdx = r * outC + oc;
                    x[dstIdx] = pooled[srcIdx];
                }
            }

            // hidden = ReLU(x * W1 + b1)
            var hidden = x.MatMul(W1); // [1,64]
            hidden = hidden.Add(b1);
            hidden = ReLUTensor(hidden, ctx);

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

        private Tensor ReLUTensor(Tensor input)
        {
            return ReLUTensor(input, null);
        }

        private Tensor ReLUTensor(Tensor input, ComputacaoContexto ctx)
        {
            var fabrica = new FabricaTensor(ctx ?? new ComputacaoCPUContexto());
            if (input is Bionix.ML.nucleo.tensor.TensorCPU srcCpu)
            {
                var outT = fabrica.Criar(srcCpu.Shape);
                for (int i = 0; i < srcCpu.Size; i++) outT[i] = Math.Max(0.0, srcCpu[i]);
                outT.RequiresGrad = true;
                outT.GradFn = new Bionix.ML.grafo.CPU.ActivationFunction(srcCpu, outT as Bionix.ML.nucleo.tensor.TensorCPU);
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
                        for (int k = 0; k < vecSize; k++) outT[i + k] = Math.Max(0.0, srcSimd[i + k]);
                    }
                    for (; i < n; i++) outT[i] = Math.Max(0.0, srcSimd[i]);
                }
                else
                {
                    for (int i = 0; i < n; i++) outT[i] = Math.Max(0.0, srcSimd[i]);
                }
                outT.RequiresGrad = true;
                outT.GradFn = new Bionix.ML.grafo.CPUSIMD.ActivationFunction(srcSimd, outT as Bionix.ML.nucleo.tensor.TensorCPUSIMD);
                return outT;
            }
            else throw new NotSupportedException("Unsupported tensor type for ReLU helper.");
        }

        public void SaveWeights(string dir)
        {
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            try { SerializadorTensor.SaveBinary(Path.Combine(dir, "w1.bin"), W1); } catch { }
            try { SerializadorTensor.SaveBinary(Path.Combine(dir, "b1.bin"), b1); } catch { }
            try { SerializadorTensor.SaveBinary(Path.Combine(dir, "w2.bin"), W2); } catch { }
            try { SerializadorTensor.SaveBinary(Path.Combine(dir, "b2.bin"), b2); } catch { }
            try { SerializadorTensor.SaveBinary(Path.Combine(dir, "convW.bin"), convW); } catch { }
            try { SerializadorTensor.SaveBinary(Path.Combine(dir, "convB.bin"), convB); } catch { }
        }

        public void LoadWeights(string dir)
        {
            try
            {
                Console.WriteLine($"[DetectorLeveB] LoadWeights: looking in '{dir}'");
                if (!Directory.Exists(dir))
                {
                    Console.WriteLine($"[DetectorLeveB] LoadWeights: directory not found: {dir}");
                    try
                    {
                        var cwd = Directory.GetCurrentDirectory();
                        Console.WriteLine($"[DetectorLeveB] CurrentDirectory: {cwd}");
                    }
                    catch { }
                    try
                    {
                        Console.WriteLine("[DetectorLeveB] Listing root directories:");
                        var roots = Directory.GetDirectories("/");
                        foreach (var r in roots) Console.WriteLine("[DetectorLeveB]   root: " + r);
                    }
                    catch { }

                    // Try alternative paths: without leading slash, or relative to current directory
                    var alt = dir.StartsWith("/") ? dir.TrimStart('/') : dir;
                    if (Directory.Exists(alt))
                    {
                        Console.WriteLine($"[DetectorLeveB] Found alternative path: {alt}");
                        dir = alt;
                    }
                    else
                    {
                        var combined = Path.Combine(Directory.GetCurrentDirectory(), alt);
                        if (Directory.Exists(combined))
                        {
                            Console.WriteLine($"[DetectorLeveB] Found combined path: {combined}");
                            dir = combined;
                        }
                        else
                        {
                            Console.WriteLine($"[DetectorLeveB] No alternative path found for {dir}");
                            return;
                        }
                    }
                }
                var files = Directory.GetFiles(dir);
                Console.WriteLine($"[DetectorLeveB] LoadWeights: found {files.Length} files");
                foreach (var f in files)
                {
                    try { var fi = new FileInfo(f); Console.WriteLine($"[DetectorLeveB]   - {fi.Name} ({fi.Length} bytes)"); } catch { }
                }

                var p = Path.Combine(dir, "w1.bin"); if (File.Exists(p) && W1 != null) { Console.WriteLine("[DetectorLeveB]   Loading w1.bin..."); var t = SerializadorTensor.LoadBinary(p); if (t != null && t.Size == W1.Size) for (int i = 0; i < W1.Size; i++) W1[i] = t[i]; }
                p = Path.Combine(dir, "b1.bin"); if (File.Exists(p) && b1 != null) { Console.WriteLine("[DetectorLeveB]   Loading b1.bin..."); var t = SerializadorTensor.LoadBinary(p); if (t != null && t.Size == b1.Size) for (int i = 0; i < b1.Size; i++) b1[i] = t[i]; }
                p = Path.Combine(dir, "w2.bin"); if (File.Exists(p) && W2 != null) { Console.WriteLine("[DetectorLeveB]   Loading w2.bin..."); var t = SerializadorTensor.LoadBinary(p); if (t != null && t.Size == W2.Size) for (int i = 0; i < W2.Size; i++) W2[i] = t[i]; }
                p = Path.Combine(dir, "b2.bin"); if (File.Exists(p) && b2 != null) { Console.WriteLine("[DetectorLeveB]   Loading b2.bin..."); var t = SerializadorTensor.LoadBinary(p); if (t != null && t.Size == b2.Size) for (int i = 0; i < b2.Size; i++) b2[i] = t[i]; }
                p = Path.Combine(dir, "convW.bin"); if (File.Exists(p) && convW != null) { Console.WriteLine("[DetectorLeveB]   Loading convW.bin..."); var t = SerializadorTensor.LoadBinary(p); if (t != null && t.Size == convW.Size) for (int i = 0; i < convW.Size; i++) convW[i] = t[i]; }
                p = Path.Combine(dir, "convB.bin"); if (File.Exists(p) && convB != null) { Console.WriteLine("[DetectorLeveB]   Loading convB.bin..."); var t = SerializadorTensor.LoadBinary(p); if (t != null && t.Size == convB.Size) for (int i = 0; i < convB.Size; i++) convB[i] = t[i]; }
                Console.WriteLine("[DetectorLeveB] LoadWeights: finished loading available files");
            }
            catch (Exception ex) { Console.WriteLine("[DetectorLeveB] LoadWeights error: " + ex.ToString()); }
        }

        public System.Collections.Generic.IEnumerable<(string name, Tensor tensor)> GetNamedParameters()
        {
            if (convW != null) yield return ("convW", convW);
            if (convB != null) yield return ("convB", convB);
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
            lock (_instLock)
            {
                Console.WriteLine($"[DetectorLeveB] GetInstance called. ctx={ctx?.GetType().Name}, pesosDir={pesosDir}");
                if (_instance == null)
                {
                    var m = new DetectorLeve();
                    m.InitializeWeights(ctx ?? new ComputacaoCPUContexto());
                    if (!string.IsNullOrEmpty(pesosDir))
                    {
                        try { Console.WriteLine($"[DetectorLeveB] Loading weights during GetInstance from {pesosDir}"); m.LoadWeights(pesosDir); } catch (Exception ex) { Console.WriteLine("[DetectorLeveB] LoadWeights threw: " + ex.ToString()); }
                    }
                    _instance = m;
                    return _instance;
                }

                // If an instance already exists but a different computation context
                // is requested, recreate the model using the requested context
                if (ctx != null && _instance._ctxUsed != null && ctx.GetType() != _instance._ctxUsed.GetType())
                {
                    var m = new DetectorLeve();
                    m.InitializeWeights(ctx);
                    if (!string.IsNullOrEmpty(pesosDir))
                    {
                        try { Console.WriteLine($"[DetectorLeveB] Recreating instance and loading weights from {pesosDir}"); m.LoadWeights(pesosDir); } catch (Exception ex) { Console.WriteLine("[DetectorLeveB] LoadWeights threw: " + ex.ToString()); }
                    }
                    _instance = m;
                }
                else if (ctx != null && _instance._ctxUsed == null)
                {
                    // existing instance was created without recording context; recreate with ctx
                    var m = new DetectorLeve();
                    m.InitializeWeights(ctx);
                    if (!string.IsNullOrEmpty(pesosDir))
                    {
                        try { Console.WriteLine($"[DetectorLeveB] Recreating instance (no prior ctx) and loading weights from {pesosDir}"); m.LoadWeights(pesosDir); } catch (Exception ex) { Console.WriteLine("[DetectorLeveB] LoadWeights threw: " + ex.ToString()); }
                    }
                    _instance = m;
                }

                return _instance;
            }
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

                // If the input crop is larger than the target 'work' on its smaller side,
                // resize the crop so its smaller side == work (preserve aspect ratio),
                // perform sliding-window on the resized image and map detections back to
                // the original crop coordinates.
                if (Math.Min(cropW, cropH) > work)
                {
                    double scale = (double)work / Math.Min(cropW, cropH);
                    int rW = Math.Max(1, (int)Math.Round(cropW * scale));
                    int rH = Math.Max(1, (int)Math.Round(cropH * scale));
                    var resized = ManipuladorDeImagem.redimensionar(crop, rW, rH);
                    // Debug: save resized image for inspection (one file per scale)
                    try
                    {
                        var outDir = Path.Combine(Environment.CurrentDirectory, "SAIDA", "debug_images");
                        Directory.CreateDirectory(outDir);
                        using var imgDbg = new Image<Rgba32>(resized.Width, resized.Height);
                        imgDbg.ProcessPixelRows(accessor =>
                        {
                            for (int yy = 0; yy < resized.Height; yy++)
                            {
                                var row = accessor.GetRowSpan(yy);
                                for (int xx = 0; xx < resized.Width; xx++)
                                {
                                    int srcIndex = (yy * resized.Width + xx) * resized.QuantidadeCanais;
                                    byte r = resized.Armazenamento[srcIndex + 0];
                                    byte g = resized.Armazenamento[srcIndex + 1];
                                    byte b = resized.Armazenamento[srcIndex + 2];
                                    row[xx] = new Rgba32(r, g, b, 255);
                                }
                            }
                        });
                        var fname = Path.Combine(outDir, $"resized_work_{work}_{DateTime.Now:yyyyMMdd_HHmmssfff}.png");
                        imgDbg.Save(fname);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("[DetectorLeveB] Failed to save debug resized image: " + ex.ToString());
                    }

                    for (int y0 = 0; y0 + work <= rH; y0 += step)
                    {
                        for (int x0 = 0; x0 + work <= rW; x0 += step)
                        {
                            var win = ManipuladorDeImagem.cortar(resized, x0, y0, work, work);
                            var t = ManipuladorDeImagem.TransformarCropParaTensorGrayscale(win, 20, ctx);
                            var outt = this.Forward(t, ctx);
                            double score = outt[0];
                            if (score < detectCutoff) continue;
                            // ignore degenerate whole-crop boxes (in resized space)
                            if (work >= Math.Max(0.99 * rW, rW) && work >= Math.Max(0.99 * rH, rH)) continue;

                            // map coordinates back to original crop space
                            int ox = Math.Max(0, (int)Math.Round(x0 / scale));
                            int oy = Math.Max(0, (int)Math.Round(y0 / scale));
                            int ow = Math.Max(1, (int)Math.Round(work / scale));
                            int oh = Math.Max(1, (int)Math.Round(work / scale));
                            per.Add((score, ox, oy, ow, oh));
                        }
                    }
                }
                else
                {
                    // input is already smaller-or-equal on the smaller side: slide on original
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
                }

                // sort per-scale and take topK
                var top = per.OrderByDescending(p => p.score).Take(topK);
                foreach (var p in top) results.Add((p.score, p.x, p.y, p.w, p.h, work));
            }
            return results;
        }

        // Aggregate detections to a single consensus bounding box where the face
        // is assumed to be the area with the highest concentration of overlapping
        // detections. Rules applied:
        // - Build a per-pixel heatmap counting how many detection rectangles cover each pixel.
        // - Find the connected component containing a max-count pixel using a threshold
        //   of 50% of max count.
        // - Keep only detections whose intersection with that component has at least
        //   `minSideFraction` (fraction of the detection side) in both width and height.
        // - If none remain, fall back to the highest-score detection.
        // - Expand final box so height >= width (grow downward), then add 20% of the
        //   square height downward, clipping to crop bounds.
        public (bool found, int x, int y, int w, int h) AggregateConsensus(System.Collections.Generic.List<(double score, int x, int y, int w, int h, int work)> detections, int cropW, int cropH, double minSideFraction = 0.4)
        {
            if (detections == null || detections.Count == 0) return (false, 0, 0, 0, 0);
            // clamp
            cropW = Math.Max(1, cropW);
            cropH = Math.Max(1, cropH);

            var heat = new int[cropW, cropH];
            int maxCount = 0;
            foreach (var d in detections)
            {
                int x0 = Math.Max(0, Math.Min(d.x, cropW - 1));
                int y0 = Math.Max(0, Math.Min(d.y, cropH - 1));
                int x1 = Math.Max(0, Math.Min(d.x + d.w, cropW));
                int y1 = Math.Max(0, Math.Min(d.y + d.h, cropH));
                for (int y = y0; y < y1; y++)
                {
                    for (int x = x0; x < x1; x++)
                    {
                        heat[x, y]++;
                        if (heat[x, y] > maxCount) maxCount = heat[x, y];
                    }
                }
            }

            if (maxCount <= 0) return (false, 0, 0, 0, 0);

            int thresh = Math.Max(1, (int)Math.Ceiling(maxCount * 0.5));
            // find a max pixel (first one) to start flood fill
            int sx = -1, sy = -1;
            for (int y = 0; y < cropH && sx < 0; y++) for (int x = 0; x < cropW; x++) if (heat[x, y] >= thresh) { sx = x; sy = y; break; }
            if (sx < 0) return (false, 0, 0, 0, 0);

            // bfs to get connected component of pixels with heat >= thresh
            var qx = new System.Collections.Generic.Queue<int>();
            var qy = new System.Collections.Generic.Queue<int>();
            var seen = new bool[cropW, cropH];
            qx.Enqueue(sx); qy.Enqueue(sy); seen[sx, sy] = true;
            int minX = sx, minY = sy, maxX = sx, maxY = sy;
            while (qx.Count > 0)
            {
                int x = qx.Dequeue(); int y = qy.Dequeue();
                minX = Math.Min(minX, x); minY = Math.Min(minY, y); maxX = Math.Max(maxX, x); maxY = Math.Max(maxY, y);
                var nbrs = new (int dx, int dy)[] { (-1,0),(1,0),(0,-1),(0,1) };
                foreach (var n in nbrs)
                {
                    int nx = x + n.dx, ny = y + n.dy;
                    if (nx < 0 || nx >= cropW || ny < 0 || ny >= cropH) continue;
                    if (seen[nx, ny]) continue;
                    if (heat[nx, ny] >= thresh)
                    {
                        seen[nx, ny] = true;
                        qx.Enqueue(nx); qy.Enqueue(ny);
                    }
                }
            }
            // component bounding box (inclusive pixels) => convert to rect coordinates
            int compX = minX, compY = minY, compW = maxX - minX + 1, compH = maxY - minY + 1;

            // filter detections by intersection fraction relative to their side
            var included = new System.Collections.Generic.List<(double score, int x, int y, int w, int h)>();
            foreach (var d in detections)
            {
                int ix0 = Math.Max(d.x, compX);
                int iy0 = Math.Max(d.y, compY);
                int ix1 = Math.Min(d.x + d.w, compX + compW);
                int iy1 = Math.Min(d.y + d.h, compY + compH);
                int iW = Math.Max(0, ix1 - ix0);
                int iH = Math.Max(0, iy1 - iy0);
                if (iW <= 0 || iH <= 0) continue;
                if (iW < Math.Ceiling(minSideFraction * d.w) || iH < Math.Ceiling(minSideFraction * d.h)) continue;
                included.Add((d.score, d.x, d.y, d.w, d.h));
            }

            int finalX, finalY, finalW, finalH;
            if (included.Count == 0)
            {
                // fallback: choose highest-score detection
                var best = detections.OrderByDescending(d => d.score).First();
                finalX = best.x; finalY = best.y; finalW = best.w; finalH = best.h;
            }
            else
            {
                finalX = included.Min(d => d.x);
                finalY = included.Min(d => d.y);
                int rx = included.Max(d => d.x + d.w);
                int ry = included.Max(d => d.y + d.h);
                finalW = rx - finalX;
                finalH = ry - finalY;
            }

            // ensure height >= width by expanding downward if needed
            if (finalH < finalW)
            {
                int desiredH = finalW;
                int extra = desiredH - finalH;
                // expand downward
                finalH = desiredH;
                // clamp within crop
                if (finalY + finalH > cropH) finalH = cropH - finalY;
            }
            // add 20% of square height downward
            int add = (int)Math.Round(0.2 * Math.Max(finalW, finalH));
            if (finalY + finalH + add > cropH) add = cropH - (finalY + finalH);
            finalH += Math.Max(0, add);

            // clip and ensure positive
            finalX = Math.Max(0, Math.Min(finalX, cropW - 1));
            finalY = Math.Max(0, Math.Min(finalY, cropH - 1));
            finalW = Math.Max(1, Math.Min(finalW, cropW - finalX));
            finalH = Math.Max(1, Math.Min(finalH, cropH - finalY));

            return (true, finalX, finalY, finalW, finalH);
        }

        // Helper: detect on a resized version to avoid sliding-window cost on huge images.
        // Resizes the input so its max dimension <= maxDim, runs DetectBest on the resized BMP
        // and maps coordinates back to the original image space.
        public (double score, int x, int y, int w, int h) DetectBestResized(BMP fullImage, ComputacaoContexto ctx = null, double detectCutoff = 0.5, int[] scales = null, int[] steps = null, int maxDim = 800)
        {
            if (fullImage == null) throw new ArgumentNullException(nameof(fullImage));
            int origW = fullImage.Width, origH = fullImage.Height;
            // If the smaller side is already less-or-equal to the target, no resize
            if (Math.Min(origW, origH) <= maxDim)
            {
                return DetectBest(fullImage, ctx, detectCutoff, scales, steps);
            }

            // Scale so the smaller side equals maxDim, keeping aspect ratio
            double scale = (double)maxDim / Math.Min(origW, origH);
            int newW = Math.Max(1, (int)Math.Round(origW * scale));
            int newH = Math.Max(1, (int)Math.Round(origH * scale));
            var resized = ManipuladorDeImagem.redimensionar(fullImage, newW, newH);
            var best = DetectBest(resized, ctx, detectCutoff, scales, steps);
            // map back
            if (double.IsNegativeInfinity(best.score) || double.IsNaN(best.score)) return best;
            int rx = Math.Max(0, (int)Math.Round(best.x / scale));
            int ry = Math.Max(0, (int)Math.Round(best.y / scale));
            int rw = Math.Max(1, (int)Math.Round(best.w / scale));
            int rh = Math.Max(1, (int)Math.Round(best.h / scale));
            return (best.score, rx, ry, rw, rh);
        }

    }
}
