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
using DetectorLeveModel;
using DetectorModel.dados;

namespace DetectorLeveModel.Runner
{
    // Local project-specific hyperparameters (kept per-project by design)
    public class HyperParameters
    {
        public int NumEpochs { get; set; } = 50;
        public int BatchSize { get; set; } = 16;
        public double InitialLearningRate { get; set; } = 1e-3;
        // if >0, limits samples processed per epoch
        public int MaxSamplesPerEpoch { get; set; } = 0;
        // if >0, training will stop early when avg epoch loss <= this threshold
        public double LossThreshold { get; set; } = 0.0;
        // if true, suppress saving per-sample outputs (.png/.txt)
        public bool SuppressOutputs { get; set; } = false;
        // if >0, training will stop early when avg epoch accuracy >= this threshold (0..1)
        public double AccuracyThreshold { get; set; } = 0.0;
    }

    public static class Program
    {
        
        public static void Main(string[] args)
        {
            var hp = new HyperParameters();
            // allow overriding via env or args
            var epochsEnv = Environment.GetEnvironmentVariable("EPOCHS");
            if (!string.IsNullOrEmpty(epochsEnv) && int.TryParse(epochsEnv, NumberStyles.Integer, CultureInfo.InvariantCulture, out var e)) hp.NumEpochs = Math.Max(1, e);
            var lossEnv = Environment.GetEnvironmentVariable("LOSS_THRESHOLD");
            if (!string.IsNullOrEmpty(lossEnv))
            {
                if (!double.TryParse(lossEnv, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var lt) && !double.TryParse(lossEnv, out lt)) lt = 0.0;
                hp.LossThreshold = Math.Max(0.0, lt);
            }
            var accEnv = Environment.GetEnvironmentVariable("ACCURACY_THRESHOLD");
            if (!string.IsNullOrEmpty(accEnv))
            {
                if (!double.TryParse(accEnv, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var at) && !double.TryParse(accEnv, out at)) at = 0.0;
                hp.AccuracyThreshold = Math.Max(0.0, Math.Min(1.0, at));
            }
            // parse simple args: --epochs N, --batch-size N, --max-samples N, --quick, --loss-threshold T, --no-outputs
            for (int ai = 0; ai < args.Length; ai++)
            {
                var a = args[ai];
                if ((a == "--epochs" || a == "-e") && ai + 1 < args.Length && int.TryParse(args[ai + 1], out var ev)) { hp.NumEpochs = Math.Max(1, ev); ai++; }
                else if ((a == "--batch-size" || a == "-b") && ai + 1 < args.Length && int.TryParse(args[ai + 1], out var bv)) { hp.BatchSize = Math.Max(1, bv); ai++; }
                else if ((a == "--max-samples" || a == "-m") && ai + 1 < args.Length && int.TryParse(args[ai + 1], out var mv)) { hp.MaxSamplesPerEpoch = Math.Max(0, mv); ai++; }
                else if (a == "--quick") { hp.MaxSamplesPerEpoch = 1; }
                else if (a == "--loss-threshold" || a == "-t") { if (ai + 1 < args.Length) { var s = args[ai+1]; if (!double.TryParse(s, NumberStyles.Float|NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var tv) && !double.TryParse(s, out tv)) tv = 0.0; hp.LossThreshold = Math.Max(0.0, tv); ai++; } }
                else if (a == "--no-outputs" || a == "--suppress-outputs") { hp.SuppressOutputs = true; }
                else if (a == "--accuracy-threshold" || a == "-a") { if (ai + 1 < args.Length) { var s = args[ai+1]; if (!double.TryParse(s, NumberStyles.Float|NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var av) && !double.TryParse(s, out av)) av = 0.0; hp.AccuracyThreshold = Math.Max(0.0, Math.Min(1.0, av)); ai++; } }
            }
            // QUICK_TEST_SAVE env for quick one-sample test (backwards compat)
            var quick = Environment.GetEnvironmentVariable("QUICK_TEST_SAVE");
            if (!string.IsNullOrEmpty(quick) && (quick == "1" || quick.Equals("true", StringComparison.OrdinalIgnoreCase))) hp.MaxSamplesPerEpoch = 1;
            var supOut = Environment.GetEnvironmentVariable("SUPPRESS_OUTPUTS");
            if (!string.IsNullOrEmpty(supOut) && (supOut == "1" || supOut.Equals("true", StringComparison.OrdinalIgnoreCase))) hp.SuppressOutputs = true;
            // also support ACCURACY_THRESHOLD env parsed with invariant or current culture
            // (accEnv handled above)
            // default MaxSamplesPerEpoch to BatchSize if unset (0)
            if (hp.MaxSamplesPerEpoch <= 0) hp.MaxSamplesPerEpoch = hp.BatchSize;

            // Enforce minimum required accuracy of 90% per user request.
            // If the configured accuracy threshold is below 0.9, raise it to 0.9 and inform the user.
            if (hp.AccuracyThreshold > 0.0 && hp.AccuracyThreshold < 0.9)
            {
                Console.WriteLine($"AccuracyThreshold too low ({hp.AccuracyThreshold:P2}), enforcing minimum 90%.");
                hp.AccuracyThreshold = 0.9;
            }
            else if (hp.AccuracyThreshold <= 0.0)
            {
                // If user didn't set an accuracy threshold, require at least 90% by default.
                Console.WriteLine("No accuracy threshold provided — enforcing minimum required accuracy = 90%.");
                hp.AccuracyThreshold = 0.9;
            }

            var computeEnv = Environment.GetEnvironmentVariable("COMPUTE") ?? "SIMD";
            ComputacaoContexto ctx = computeEnv.Equals("CPU", StringComparison.OrdinalIgnoreCase) ? (ComputacaoContexto)new ComputacaoCPUContexto() : new ComputacaoCPUSIMDContexto();
            try
            {
                Console.WriteLine("DetectorLeveModel Runner: initializing context");
                var model = new DetectorLeve();
                model.InitializeWeights(ctx);

                // Resolve dataset similar to Detector runner
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
                    // try other csv names
                    var alt = Directory.GetFiles(datasetRoot, "list_landmarks*.csv").FirstOrDefault();
                    if (alt != null) annotations = alt;
                }
                if (!File.Exists(annotations) || !Directory.Exists(imagesFolder)) { Console.WriteLine("Annotations or images folder missing."); return; }

                var loader = new DataLoader(annotations, imagesFolder);

                var fabrica = new FabricaTensor(ctx);
                var bceFn = FabricaFuncoesPerda.CriarBCE(ctx);

                // collect model params
                var paramList = new List<Tensor>();
                foreach (var kv in model.GetNamedParameters()) if (kv.tensor != null) paramList.Add(kv.tensor);
                var optimizer = FabricaOtimizadores.CriarStatefulSGD(paramList, ctx, lr: hp.InitialLearningRate, momentum: 0.9);

                var rnd = new Random(123);
                var saidaDir = Path.Combine(Directory.GetCurrentDirectory(), "SAIDA"); Directory.CreateDirectory(saidaDir);
                var pesosDirRoot = Path.Combine(Directory.GetCurrentDirectory(), "PESOS", "CLASSIFICADOR_DETECTOR_LEVE"); Directory.CreateDirectory(pesosDirRoot);

                var anns = loader.ReadAnnotations().ToList();
                if (anns.Count == 0) { Console.WriteLine("No annotations found."); return; }

                for (int epoch = 0; epoch < hp.NumEpochs; epoch++)
                {
                    Console.WriteLine($"Epoch {epoch}");
                    // shuffle
                    anns = anns.OrderBy(x => rnd.Next()).ToList();
                    int take = hp.MaxSamplesPerEpoch > 0 ? Math.Min(hp.MaxSamplesPerEpoch, anns.Count) : anns.Count;
                    int processed = 0;
                    double epochLoss = 0.0; int lossCount = 0;
                    int correctPos = 0, correctNeg = 0; int totalPredictions = 0;
                    for (int i = 0; i < take; i++)
                    {
                        var ann = anns[i];
                        if (ann.Boxes == null || ann.Boxes.Count == 0) continue;
                        try
                        {
                            // load image bmp
                            var bmp = ManipuladorDeImagem.carregarBmpDeJPEG(ann.ImagePath);
                            // use first box
                            var gb = ann.Boxes[0];
                            var crop = ManipuladorDeImagem.CropSquare(bmp, gb.X, gb.Y, gb.Width, gb.Height);
                            var posTensor = ManipuladorDeImagem.TransformarCropParaTensorGrayscale(crop, 20, ctx);

                            // negative: sample random square of same side
                            int side = Math.Max(gb.Width, gb.Height);
                            // ensure side within image
                            side = Math.Min(side, bmp.Width);
                            side = Math.Min(side, bmp.Height);
                            int attempts = 0; BMP negCrop = null;
                            while (attempts < 20)
                            {
                                attempts++;
                                int nx = rnd.Next(0, Math.Max(1, bmp.Width - side));
                                int ny = rnd.Next(0, Math.Max(1, bmp.Height - side));
                                // check IOU with gb
                                var iou = ComputeIoU(nx, ny, side, side, gb.X, gb.Y, gb.Width, gb.Height);
                                if (iou < 0.1)
                                {
                                    negCrop = ManipuladorDeImagem.cortar(bmp, nx, ny, side, side);
                                    break;
                                }
                            }
                            if (negCrop == null) negCrop = ManipuladorDeImagem.CropSquare(bmp, gb.X + side, gb.Y + side, side, side);
                            var negTensor = ManipuladorDeImagem.TransformarCropParaTensorGrayscale(negCrop, 20, ctx);

                            // Forward positive
                            var outPos = model.Forward(posTensor, ctx); // [1,1]
                            var outNeg = model.Forward(negTensor, ctx);

                            // targets
                            var tPos = fabrica.FromArray(new int[] { 1, 1 }, new double[] { 1.0 });
                            var tNeg = fabrica.FromArray(new int[] { 1, 1 }, new double[] { 0.0 });

                            var lossPos = bceFn(outPos, tPos);
                            var lossNeg = bceFn(outNeg, tNeg);
                            var total = lossPos.Add(lossNeg);
                            epochLoss += total[0]; lossCount++;
                            // accuracy counters
                            var pOutVal = outPos[0]; var nOutVal = outNeg[0];
                            if (pOutVal > 0.5) correctPos++;
                            if (nOutVal <= 0.5) correctNeg++;
                            totalPredictions += 2;

                            total.Backward();
                            optimizer.Step();

                            // save outputs: resized crops to SAIDA with probs (skippable)
                            try
                            {
                                var pOut = outPos[0]; var nOut = outNeg[0];
                                if (!hp.SuppressOutputs)
                                {
                                    var baseP = Path.Combine(saidaDir, $"ep{epoch:00}_idx{i:0000}_pos_{pOut:F4}.png");
                                    var baseN = Path.Combine(saidaDir, $"ep{epoch:00}_idx{i:0000}_neg_{nOut:F4}.png");
                                    // save using ImageSharp: create image from resized BMP
                                    var imgP = ToImage(ManipuladorDeImagem.redimensionar(crop, 20, 20));
                                    using (var fsP = File.Create(baseP)) imgP.Save(fsP, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                                    var infoPpath = Path.ChangeExtension(baseP, ".txt");
                                    var predP = pOut > 0.5 ? "positive" : "negative";
                                    File.WriteAllText(infoPpath, $"label: positive\npred: {predP}\nprob: {pOut:F6}");
                                    var imgN = ToImage(ManipuladorDeImagem.redimensionar(negCrop, 20, 20));
                                    using (var fsN = File.Create(baseN)) imgN.Save(fsN, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                                    var infoNpath = Path.ChangeExtension(baseN, ".txt");
                                    var predN = nOut > 0.5 ? "positive" : "negative";
                                    File.WriteAllText(infoNpath, $"label: negative\npred: {predN}\nprob: {nOut:F6}");
                                }
                            }
                            catch { }

                            processed++;
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Sample error: {ex.Message}");
                        }
                    }

                    var avgLoss = (lossCount>0?epochLoss/lossCount:double.NaN);
                    var accuracy = totalPredictions > 0 ? (double)(correctPos + correctNeg) / totalPredictions : double.NaN;
                    Console.WriteLine($"Epoch {epoch} processed {processed} samples, avg loss = {avgLoss:F6}, accuracy = {accuracy:P2}");

                    // checkpoint save
                    try
                    {
                        // save model and optimizer state
                        var tmp = pesosDirRoot + ".tmp_" + DateTime.UtcNow.Ticks;
                        if (Directory.Exists(tmp)) Directory.Delete(tmp, true);
                        Directory.CreateDirectory(tmp);
                        model.SaveWeights(tmp);
                        optimizer?.SaveState(tmp);
                        var metaObj = new { epoch = epoch, lr = optimizer?.Lr ?? hp.InitialLearningRate, timestamp = DateTime.UtcNow };
                        File.WriteAllText(Path.Combine(tmp, "meta.json"), System.Text.Json.JsonSerializer.Serialize(metaObj));
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
                    // early stopping by accuracy threshold
                    if (hp.AccuracyThreshold > 0.0 && totalPredictions > 0)
                    {
                        var accVal = (double)(correctPos + correctNeg) / totalPredictions;
                        if (accVal >= hp.AccuracyThreshold)
                        {
                            Console.WriteLine($"Early stopping: accuracy {accVal:P2} >= threshold {hp.AccuracyThreshold:P2}");
                            break;
                        }
                    }
                }
            }
            finally { if (ctx is IDisposable d) d.Dispose(); }
        }

        // helper IoU for axis-aligned squares
        private static double ComputeIoU(int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2)
        {
            int xa = Math.Max(x1, x2);
            int ya = Math.Max(y1, y2);
            int xb = Math.Min(x1 + w1, x2 + w2);
            int yb = Math.Min(y1 + h1, y2 + h2);
            int interW = xb - xa; int interH = yb - ya;
            if (interW <= 0 || interH <= 0) return 0.0;
            double inter = interW * interH;
            double union = w1 * h1 + w2 * h2 - inter;
            return inter / Math.Max(1.0, union);
        }

        // small helper to access private ToImage and redimensionar from ManipuladorDeImagem via reflection
        private static SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32> ToImage(BMP bmp)
        {
            // use existing private method via reflection
            var t = typeof(ManipuladorDeImagem).GetMethod("ToImage", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
            if (t != null) return (SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32>)t.Invoke(null, new object[] { bmp });
            // fallback: construct image
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
