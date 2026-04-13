using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Bionix.ML.computacao;
using Bionix.ML.dados.imagem;
using Bionix.ML.dados.imagem.bmp;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.nucleo.otimizadores;
using Bionix.ML.nucleo.funcoesPerda;
using Bionix.ML.dados.serializacao;
using System.Globalization;
using IdentificadorLeveModel;
using DetectorModel.dados;

namespace IdentificadorLeveModel.Runner
{
    public class HyperParameters
    {
        public int NumEpochs { get; set; } = 1000;
        public int BatchSize { get; set; } = 16;
        public double InitialLearningRate { get; set; } = 1e-3;
        public double MinLearningRate { get; set; } = 1e-6;
        public bool ReduceOnPlateau { get; set; } = true;
        public int ReduceOnPlateauPatience { get; set; } = 8;
        public double ReduceOnPlateauFactor { get; set; } = 0.5;
        public int MaxSamplesPerEpoch { get; set; } = 0;
        public double LossThreshold { get; set; } = 0.0;
        public bool SuppressOutputs { get; set; } = false;
    }

    public static class Program
    {
        public static void Main(string[] args)
        {
            var hp = new HyperParameters();
            bool resume = false;
            var epochsEnv = Environment.GetEnvironmentVariable("EPOCHS");
            if (!string.IsNullOrEmpty(epochsEnv) && int.TryParse(epochsEnv, NumberStyles.Integer, CultureInfo.InvariantCulture, out var e)) hp.NumEpochs = Math.Max(1, e);
            var supOut = Environment.GetEnvironmentVariable("SUPPRESS_OUTPUTS");
            if (!string.IsNullOrEmpty(supOut) && (supOut == "1" || supOut.Equals("true", StringComparison.OrdinalIgnoreCase))) hp.SuppressOutputs = true;

            var computeEnv = Environment.GetEnvironmentVariable("COMPUTE") ?? "SIMD";
            ComputacaoContexto ctx = computeEnv.Equals("CPU", StringComparison.OrdinalIgnoreCase) ? (ComputacaoContexto)new ComputacaoCPUContexto() : new ComputacaoCPUSIMDContexto();

            try
            {
                Console.WriteLine("IdentificadorLeve Runner: initializing context");
                var model = new IdentificadorLeve();
                model.InitializeWeights(ctx);

                // dataset resolution (same as detector runner)
                var initialCwd = Directory.GetCurrentDirectory();
                string datasetRoot = null;
                var cand1 = Path.Combine(initialCwd, "BIONIX-ML-VIVAZ", "DATASET", "faces_com_landmarks");
                var cand2 = Path.Combine(initialCwd, "BIONIX-ML", "DATASET", "faces_com_landmarks");
                var cand3 = Path.Combine(initialCwd, "DATASET", "faces_com_landmarks");
                if (Directory.Exists(cand1)) datasetRoot = cand1; else if (Directory.Exists(cand2)) datasetRoot = cand2; else if (Directory.Exists(cand3)) datasetRoot = cand3;
                if (datasetRoot == null) { Console.WriteLine("Dataset faces_com_landmarks not found."); return; }

                var annotations = Path.Combine(datasetRoot, "list_landmarks_align_celeba.csv");
                var imagesFolder = Path.Combine(datasetRoot, "img_align_celeba", "img_align_celeba");
                if (!File.Exists(annotations)) {
                    var alt = Directory.GetFiles(datasetRoot, "list_landmarks*.csv").FirstOrDefault();
                    if (alt != null) annotations = alt;
                }
                if (!File.Exists(annotations) || !Directory.Exists(imagesFolder)) { Console.WriteLine("Annotations or images folder missing."); return; }

                var loader = new DataLoader(annotations, imagesFolder);
                var anns = loader.ReadAnnotations().ToList();
                if (anns.Count == 0) { Console.WriteLine("No annotations found."); return; }

                var fabrica = new FabricaTensor(ctx);
                var bceFn = FabricaFuncoesPerda.CriarBCE(ctx);

                int embeddingSize = model.EmbeddingSize;
                int hidden = 64;

                // comparator weights: emb->hidden for each branch, bias, and final out
                var Wa = fabrica.Criar(embeddingSize, hidden); Wa.RequiresGrad = true;
                var Wb = fabrica.Criar(embeddingSize, hidden); Wb.RequiresGrad = true;
                var bh = fabrica.Criar(1, hidden); bh.RequiresGrad = true;
                var Wo = fabrica.Criar(hidden, 1); Wo.RequiresGrad = true;
                var bo = fabrica.Criar(1, 1); bo.RequiresGrad = true;
                var rnd = new Random(123);
                foreach (var t in new Tensor[] { Wa, Wb, Wo }) for (int i = 0; i < t.Size; i++) t[i] = (rnd.NextDouble() - 0.5) * 0.01;
                foreach (var t in new Tensor[] { bh, bo }) for (int i = 0; i < t.Size; i++) t[i] = 0.0;

                // collect parameters (model params + comparator params)
                var paramList = new List<Tensor>();
                foreach (var kv in model.GetNamedParameters()) if (kv.tensor != null) paramList.Add(kv.tensor);
                paramList.Add(Wa); paramList.Add(Wb); paramList.Add(bh); paramList.Add(Wo); paramList.Add(bo);

                var optimizer = FabricaOtimizadores.CriarStatefulSGD(paramList, ctx, lr: hp.InitialLearningRate, momentum: 0.9);

                var saidaDir = Path.Combine(Directory.GetCurrentDirectory(), "SAIDA"); Directory.CreateDirectory(saidaDir);
                var pesosDirRoot = Path.Combine(Directory.GetCurrentDirectory(), "PESOS", "IDENTIFICADOR_LEVE"); Directory.CreateDirectory(pesosDirRoot);

                // training loop
                if (hp.MaxSamplesPerEpoch <= 0) hp.MaxSamplesPerEpoch = hp.BatchSize;

                for (int epoch = 0; epoch < hp.NumEpochs; epoch++)
                {
                    Console.WriteLine($"Epoch {epoch}");
                    anns = anns.OrderBy(x => rnd.Next()).ToList();
                    int take = Math.Min(hp.MaxSamplesPerEpoch > 0 ? hp.MaxSamplesPerEpoch : anns.Count, anns.Count);
                    double epochLoss = 0.0; int lossCount = 0;

                    for (int i = 0; i < take; i++)
                    {
                        var ann = anns[i];
                        if (ann.Boxes == null || ann.Boxes.Count == 0) continue;
                        try
                        {
                            var bmp = ManipuladorDeImagem.carregarBmpDeJPEG(ann.ImagePath);
                            var gb = ann.Boxes[0];
                            // base square crop
                            var crop1 = ManipuladorDeImagem.CropSquare(bmp, gb.X, gb.Y, gb.Width, gb.Height);
                            var cropPos1 = ManipuladorDeImagem.redimensionar(crop1, model.InputSide, model.InputSide);

                            // positive variant: jitter center within +/-10% of side
                            int side = Math.Max(gb.Width, gb.Height);
                            double jitter = 0.10; // 10%
                            int maxJ = (int)Math.Round(jitter * side);
                            int dx = rnd.Next(-maxJ, maxJ + 1);
                            int dy = rnd.Next(-maxJ, maxJ + 1);
                            int cx = gb.X + dx; int cy = gb.Y + dy;
                            var crop2 = ManipuladorDeImagem.CropSquare(bmp, cx, cy, side, side);
                            var cropPos2 = ManipuladorDeImagem.redimensionar(crop2, model.InputSide, model.InputSide);

                            // negative: pick another random annotation (different image)
                            int attempts = 0; BMP negCropBmp = null;
                            while (attempts < 20)
                            {
                                attempts++;
                                var other = anns[rnd.Next(anns.Count)];
                                if (other.ImagePath == ann.ImagePath) continue;
                                var otherBmp = ManipuladorDeImagem.carregarBmpDeJPEG(other.ImagePath);
                                // random square within other image
                                int s = Math.Min(Math.Min(otherBmp.Width, otherBmp.Height), side);
                                int nx = rnd.Next(0, Math.Max(1, otherBmp.Width - s));
                                int ny = rnd.Next(0, Math.Max(1, otherBmp.Height - s));
                                negCropBmp = ManipuladorDeImagem.cortar(otherBmp, nx, ny, s, s);
                                if (negCropBmp != null) break;
                            }
                            if (negCropBmp == null) negCropBmp = ManipuladorDeImagem.CropSquare(bmp, gb.X + side, gb.Y + side, side, side);
                            var negResized = ManipuladorDeImagem.redimensionar(negCropBmp, model.InputSide, model.InputSide);

                            // tensors
                            var tPos1 = ManipuladorDeImagem.TransformarCropParaTensorGrayscale(cropPos1, model.InputSide, ctx);
                            var tPos2 = ManipuladorDeImagem.TransformarCropParaTensorGrayscale(cropPos2, model.InputSide, ctx);
                            var tNeg = ManipuladorDeImagem.TransformarCropParaTensorGrayscale(negResized, model.InputSide, ctx);

                            // embeddings: each is [1,emb]
                            var embPos1 = model.Forward(tPos1, ctx);
                            var embPos2 = model.Forward(tPos2, ctx);
                            var embNeg = model.Forward(tNeg, ctx);

                            // comparator forward
                            var hPos = embPos1.MatMul(Wa).Add(embPos2.MatMul(Wb)).Add(bh);
                            hPos = Sigmoid(hPos, ctx);
                            var outPos = hPos.MatMul(Wo).Add(bo);
                            outPos = Sigmoid(outPos, ctx);

                            var hNeg = embPos1.MatMul(Wa).Add(embNeg.MatMul(Wb)).Add(bh);
                            hNeg = Sigmoid(hNeg, ctx);
                            var outNeg = hNeg.MatMul(Wo).Add(bo);
                            outNeg = Sigmoid(outNeg, ctx);

                            // targets
                            var targPos = fabrica.FromArray(new int[] { 1, 1 }, new double[] { 1.0 });
                            var targNeg = fabrica.FromArray(new int[] { 1, 1 }, new double[] { 0.0 });

                            var lossPos = bceFn(outPos, targPos);
                            var lossNeg = bceFn(outNeg, targNeg);
                            var total = lossPos.Add(lossNeg);
                            epochLoss += total[0]; lossCount++;

                            total.Backward();
                            optimizer.Step();

                            // optional outputs: save cropped pairs occasionally
                            try
                            {
                                if (!hp.SuppressOutputs && (i % 100 == 0))
                                {
                                    var idxStr = i.ToString("0000");
                                    var p1 = Path.Combine(saidaDir, $"ep{epoch:000}_{idxStr}_pos1.png");
                                    using (var fs = File.Create(p1)) ToImage(cropPos1).Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                                    var p2 = Path.Combine(saidaDir, $"ep{epoch:000}_{idxStr}_pos2.png");
                                    using (var fs = File.Create(p2)) ToImage(cropPos2).Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                                    var pn = Path.Combine(saidaDir, $"ep{epoch:000}_{idxStr}_neg.png");
                                    using (var fs = File.Create(pn)) ToImage(negResized).Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                                }
                            }
                            catch { }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Sample error: {ex.Message}");
                        }
                    }

                    var avgLoss = (lossCount > 0 ? epochLoss / lossCount : double.NaN);
                    Console.WriteLine($"Epoch {epoch} avg loss = {avgLoss:F6}");

                    // checkpoint
                    try
                    {
                        var tmp = pesosDirRoot + ".tmp_" + DateTime.UtcNow.Ticks;
                        if (Directory.Exists(tmp)) Directory.Delete(tmp, true);
                        Directory.CreateDirectory(tmp);
                        model.SaveWeights(tmp);
                        try { SerializadorTensor.SaveBinary(Path.Combine(tmp, "comp_Wa.bin"), Wa); } catch { }
                        try { SerializadorTensor.SaveBinary(Path.Combine(tmp, "comp_Wb.bin"), Wb); } catch { }
                        try { SerializadorTensor.SaveBinary(Path.Combine(tmp, "comp_bh.bin"), bh); } catch { }
                        try { SerializadorTensor.SaveBinary(Path.Combine(tmp, "comp_Wo.bin"), Wo); } catch { }
                        try { SerializadorTensor.SaveBinary(Path.Combine(tmp, "comp_bo.bin"), bo); } catch { }
                        if (Directory.Exists(pesosDirRoot)) Directory.Delete(pesosDirRoot, true);
                        Directory.Move(tmp, pesosDirRoot);
                        Console.WriteLine($"Checkpoint saved to {pesosDirRoot}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Checkpoint failed: {ex.Message}");
                    }
                    // early stopping by loss threshold
                    if (hp.LossThreshold > 0.0 && lossCount > 0)
                    {
                        if (avgLoss <= hp.LossThreshold)
                        {
                            Console.WriteLine($"Early stopping: avg loss {avgLoss:F6} <= threshold {hp.LossThreshold:F6}");
                            break;
                        }
                    }
                }
            }
            finally { if (ctx is IDisposable d) d.Dispose(); }
        }

        private static Tensor Sigmoid(Tensor input, ComputacaoContexto ctx)
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

        // small helper to convert BMP to ImageSharp Image for saving
        private static SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32> ToImage(BMP bmp)
        {
            var t = typeof(ManipuladorDeImagem).GetMethod("ToImage", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
            if (t != null) return (SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32>)t.Invoke(null, new object[] { bmp });
            var img = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32>(bmp.Width, bmp.Height);
            img.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < bmp.Height; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < bmp.Width; x++)
                    {
                        int idx = (y * bmp.Width + x) * bmp.QuantidadeCanais;
                        byte r = bmp.Armazenamento[idx + 0];
                        byte g = bmp.Armazenamento[idx + 1];
                        byte b = bmp.Armazenamento[idx + 2];
                        row[x] = new SixLabors.ImageSharp.PixelFormats.Rgba32(r, g, b, 255);
                    }
                }
            });
            return img;
        }
    }
}

