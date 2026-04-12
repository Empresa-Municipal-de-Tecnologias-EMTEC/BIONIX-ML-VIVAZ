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
        public int NumEpochs { get; set; } = 1000;
        public int BatchSize { get; set; } = 16;
        public double InitialLearningRate { get; set; } = 1e-3;
        // minimum learning rate (for schedulers)
        public double MinLearningRate { get; set; } = 1e-6;
        // reduce-on-plateau schedule
        public bool ReduceOnPlateau { get; set; } = true;
        public int ReduceOnPlateauPatience { get; set; } = 8;
        public double ReduceOnPlateauFactor { get; set; } = 0.5;
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
                bool resume = false;
            string optimizerName = "sgd";
            double gradClip = 0.0;
            double weightDecay = 0.0;
            bool detectTest = false;
            int detectSamples = 20;
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
                else if (a == "--resume") { resume = true; }
                else if (a == "--optimizer" && ai + 1 < args.Length) { optimizerName = args[ai+1].ToLowerInvariant(); ai++; }
                else if (a == "--detect-test") { detectTest = true; }
                else if (a == "--detect-samples" && ai + 1 < args.Length && int.TryParse(args[ai+1], out var dsv)) { detectSamples = Math.Max(1, dsv); ai++; }
                else if (a == "--loss-threshold" || a == "-t") { if (ai + 1 < args.Length) { var s = args[ai+1]; if (!double.TryParse(s, NumberStyles.Float|NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var tv) && !double.TryParse(s, out tv)) tv = 0.0; hp.LossThreshold = Math.Max(0.0, tv); ai++; } }
                else if (a == "--no-outputs" || a == "--suppress-outputs") { hp.SuppressOutputs = true; }
                else if (a == "--accuracy-threshold" || a == "-a") { if (ai + 1 < args.Length) { var s = args[ai+1]; if (!double.TryParse(s, NumberStyles.Float|NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var av) && !double.TryParse(s, out av)) av = 0.0; hp.AccuracyThreshold = Math.Max(0.0, Math.Min(1.0, av)); ai++; } }
            }
            // QUICK_TEST_SAVE env for quick one-sample test (backwards compat)
            var quick = Environment.GetEnvironmentVariable("QUICK_TEST_SAVE");
            if (!string.IsNullOrEmpty(quick) && (quick == "1" || quick.Equals("true", StringComparison.OrdinalIgnoreCase))) hp.MaxSamplesPerEpoch = 1;
            var supOut = Environment.GetEnvironmentVariable("SUPPRESS_OUTPUTS");
            if (!string.IsNullOrEmpty(supOut) && (supOut == "1" || supOut.Equals("true", StringComparison.OrdinalIgnoreCase))) hp.SuppressOutputs = true;
            var lrEnv = Environment.GetEnvironmentVariable("INITIAL_LR") ?? Environment.GetEnvironmentVariable("LR");
            if (!string.IsNullOrEmpty(lrEnv)) { if (!double.TryParse(lrEnv, NumberStyles.Float|NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var lrv) && !double.TryParse(lrEnv, out lrv)) lrv = hp.InitialLearningRate; hp.InitialLearningRate = Math.Max(1e-12, lrv); }
            var minLrEnv = Environment.GetEnvironmentVariable("MIN_LR");
            if (!string.IsNullOrEmpty(minLrEnv) && (double.TryParse(minLrEnv, NumberStyles.Float|NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var mlr) || double.TryParse(minLrEnv, out mlr))) hp.MinLearningRate = Math.Max(1e-12, mlr);
            var ropEnv = Environment.GetEnvironmentVariable("REDUCE_ON_PLATEAU");
            if (!string.IsNullOrEmpty(ropEnv) && (ropEnv == "0" || ropEnv.Equals("false", StringComparison.OrdinalIgnoreCase))) hp.ReduceOnPlateau = false;
            var patEnv = Environment.GetEnvironmentVariable("ROP_PATIENCE");
            if (!string.IsNullOrEmpty(patEnv) && int.TryParse(patEnv, out var patv)) hp.ReduceOnPlateauPatience = Math.Max(1, patv);
            var facEnv = Environment.GetEnvironmentVariable("ROP_FACTOR");
            if (!string.IsNullOrEmpty(facEnv) && (double.TryParse(facEnv, NumberStyles.Float|NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var facv) || double.TryParse(facEnv, out facv))) hp.ReduceOnPlateauFactor = Math.Max(0.0, Math.Min(1.0, facv));
            var optEnv = Environment.GetEnvironmentVariable("OPTIMIZER");
            if (!string.IsNullOrEmpty(optEnv)) optimizerName = optEnv.ToLowerInvariant();
            var gcEnv = Environment.GetEnvironmentVariable("GRAD_CLIP");
            if (!string.IsNullOrEmpty(gcEnv) && double.TryParse(gcEnv, NumberStyles.Float|NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var gcv)) gradClip = Math.Max(0.0, gcv);
            var wdEnv = Environment.GetEnvironmentVariable("WEIGHT_DECAY");
            if (!string.IsNullOrEmpty(wdEnv) && double.TryParse(wdEnv, NumberStyles.Float|NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var wdv)) weightDecay = Math.Max(0.0, wdv);
            // allow disabling the enforced minimum accuracy (set to "0" or "false" to disable)
            var enforceAccEnv = Environment.GetEnvironmentVariable("ENFORCE_MIN_ACCURACY");
            bool enforceMinAccuracy = true;
            if (!string.IsNullOrEmpty(enforceAccEnv) && (enforceAccEnv == "0" || enforceAccEnv.Equals("false", StringComparison.OrdinalIgnoreCase))) enforceMinAccuracy = false;
            // also support ACCURACY_THRESHOLD env parsed with invariant or current culture
            // (accEnv handled above)
            // default MaxSamplesPerEpoch to BatchSize if unset (0)
            if (hp.MaxSamplesPerEpoch <= 0) hp.MaxSamplesPerEpoch = hp.BatchSize;

            // Enforce minimum required accuracy of 90% per user request unless disabled explicitly.
            if (enforceMinAccuracy)
            {
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
            }
            else
            {
                Console.WriteLine("ENFORCE_MIN_ACCURACY=0 detected — skipping enforced 90% accuracy requirement.");
            }

            var computeEnv = Environment.GetEnvironmentVariable("COMPUTE") ?? "SIMD";
            ComputacaoContexto ctx = computeEnv.Equals("CPU", StringComparison.OrdinalIgnoreCase) ? (ComputacaoContexto)new ComputacaoCPUContexto() : new ComputacaoCPUSIMDContexto();
            
            // simple Reduce-on-Plateau scheduler (kept local to runner)
            ReduceOnPlateauScheduler scheduler = null;
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
                if (optimizerName == "adam" || optimizerName == "adamw")
                {
                    Console.WriteLine($"Optimizer requested: {optimizerName} — Adam not available in this runtime, falling back to SGD with momentum.");
                }
                var optimizer = FabricaOtimizadores.CriarStatefulSGD(paramList, ctx, lr: hp.InitialLearningRate, momentum: 0.9);
                // initialize scheduler
                if (hp.ReduceOnPlateau)
                {
                    scheduler = new ReduceOnPlateauScheduler(hp.InitialLearningRate, hp.MinLearningRate, hp.ReduceOnPlateauFactor, hp.ReduceOnPlateauPatience);
                }
                
                

                var rnd = new Random(123);
                var saidaDir = Path.Combine(Directory.GetCurrentDirectory(), "SAIDA"); Directory.CreateDirectory(saidaDir);
                var pesosDirRoot = Path.Combine(Directory.GetCurrentDirectory(), "PESOS", "CLASSIFICADOR_DETECTOR_LEVE"); Directory.CreateDirectory(pesosDirRoot);
                var csvPath = Path.Combine(pesosDirRoot, "training_log.csv");
                if (!File.Exists(csvPath)) File.WriteAllText(csvPath, "epoch,avgLoss,accuracy,lr\n");

                // If requested, attempt to resume from existing checkpoint (load weights and optimizer state)
                if (resume)
                {
                    try
                    {
                        if (Directory.Exists(pesosDirRoot))
                        {
                            model.LoadWeights(pesosDirRoot);
                            optimizer?.LoadState(pesosDirRoot);
                            // try to load meta.json and restore scheduler state and lr
                            var metaPath = Path.Combine(pesosDirRoot, "meta.json");
                            if (File.Exists(metaPath))
                            {
                                try
                                {
                                    using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(metaPath));
                                    var root = doc.RootElement;
                                    if (root.TryGetProperty("lr", out var lrEl) && lrEl.ValueKind == System.Text.Json.JsonValueKind.Number)
                                    {
                                        var savedLr = lrEl.GetDouble();
                                        if (optimizer != null) optimizer.Lr = savedLr;
                                        if (scheduler != null) scheduler.CurrentLr = savedLr;
                                    }
                                    if (root.TryGetProperty("scheduler", out var schedEl) && schedEl.ValueKind == System.Text.Json.JsonValueKind.Object && scheduler != null)
                                    {
                                        if (schedEl.TryGetProperty("bestLoss", out var bEl) && bEl.ValueKind == System.Text.Json.JsonValueKind.Number) scheduler.BestLoss = bEl.GetDouble();
                                        if (schedEl.TryGetProperty("epochsSinceImprovement", out var eEl) && eEl.ValueKind == System.Text.Json.JsonValueKind.Number) scheduler.EpochsSinceImprovement = eEl.GetInt32();
                                        if (schedEl.TryGetProperty("currentLr", out var cEl) && cEl.ValueKind == System.Text.Json.JsonValueKind.Number) scheduler.CurrentLr = cEl.GetDouble();
                                    }
                                }
                                catch { }
                            }
                            Console.WriteLine($"Resumed model and optimizer state from {pesosDirRoot}");
                            // print simple diagnostics: L2 norm of each parameter
                            try
                            {
                                int pi = 0;
                                foreach (var p in paramList)
                                {
                                    if (p == null) continue;
                                    double sumsq = 0.0;
                                    for (int ii = 0; ii < p.Size; ii++) { var v = p[ii]; sumsq += v * v; }
                                    double l2 = Math.Sqrt(sumsq);
                                    var shapeStr = p.Shape != null ? $"[{string.Join(',', p.Shape)}]" : "[]";
                                    Console.WriteLine($"Param[{pi}] shape={shapeStr} L2={l2:E6}");
                                    pi++;
                                }
                            }
                            catch { }
                        }
                        else
                        {
                            Console.WriteLine($"Resume requested but checkpoint not found at {pesosDirRoot}");
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Resume failed: {ex.Message}");
                    }
                }

                var anns = loader.ReadAnnotations().ToList();
                if (anns.Count == 0) { Console.WriteLine("No annotations found."); return; }

                // If detect-test mode requested, run detection tests and exit
                if (detectTest)
                {
                    Console.WriteLine($"Running detect-test on {detectSamples} samples...");
                    var detModel = DetectorLeveModel.DetectorLeve.GetInstance(ctx, pesosDirRoot);
                    var sampleList = anns.Where(a => a.Boxes != null && a.Boxes.Count > 0).OrderBy(x => rnd.Next()).Take(detectSamples).ToList();
                    int idx = 0;
                    foreach (var ann in sampleList)
                    {
                        try
                        {
                            var bmpFull = ManipuladorDeImagem.carregarBmpDeJPEG(ann.ImagePath);
                            var gb = ann.Boxes[0];
                            // pick random expansion factor p in [0.5, 1.5]
                            double p = rnd.NextDouble() * (1.5 - 0.5) + 0.5;
                            // diagonal of the annotated box
                            double diag = Math.Sqrt((double)gb.Width * gb.Width + (double)gb.Height * gb.Height);
                            // expand by delta = p * diag on both corners (grow along diagonal)
                            double delta = p * diag;
                            double x0d = gb.X - delta;
                            double y0d = gb.Y - delta;
                            double x1d = gb.X + gb.Width + delta;
                            double y1d = gb.Y + gb.Height + delta;
                            // clamp negatives to zero and max to image bounds
                            int x0 = (int)Math.Max(0, Math.Floor(x0d));
                            int y0 = (int)Math.Max(0, Math.Floor(y0d));
                            int x1 = (int)Math.Min(bmpFull.Width, Math.Ceiling(x1d));
                            int y1 = (int)Math.Min(bmpFull.Height, Math.Ceiling(y1d));
                            int w = Math.Max(1, x1 - x0);
                            int h = Math.Max(1, y1 - y0);
                            var crop = ManipuladorDeImagem.cortar(bmpFull, x0, y0, w, h);

                            // perform multi-scale sliding-window detection on the crop using the model
                            int[] scales = new int[] { 32, 48, 64 };
                            int[] stepsArr = new int[] { 4, 6, 8 };
                            double bestScore = double.NegativeInfinity;
                            int bestX = 0, bestY = 0, bestW = 0, bestH = 0;
                            for (int si = 0; si < scales.Length; si++)
                            {
                                int work = scales[si]; int step = stepsArr[si];
                                int minSideCrop = Math.Min(crop.Width, crop.Height);
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
                                        var outt = model.Forward(t, ctx);
                                        double score = outt[0];
                                        if (score > bestScore)
                                        {
                                            bestScore = score;
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
                                            bestX = bx; bestY = by; bestW = bw; bestH = bh;
                                        }
                                    }
                                }
                            }

                            // convert crop to image and draw detection rectangle (in crop coords)
                            var img = ToImage(crop);
                            if (!double.IsNegativeInfinity(bestScore))
                            {
                                int dx = Math.Max(0, Math.Min(bestX, crop.Width - 1));
                                int dy = Math.Max(0, Math.Min(bestY, crop.Height - 1));
                                int dw = Math.Min(bestW, crop.Width - dx);
                                int dh = Math.Min(bestH, crop.Height - dy);
                                var color = new SixLabors.ImageSharp.PixelFormats.Rgba32(255, 0, 0, 255);
                                img.ProcessPixelRows(accessor =>
                                {
                                    for (int y = dy; y < dy + dh; y++)
                                    {
                                        if (y < 0 || y >= img.Height) continue;
                                        for (int x = dx; x < dx + dw; x++)
                                        {
                                            if (x < 0 || x >= img.Width) continue;
                                            bool border = (x - dx < 2) || (dx + dw - 1 - x < 2) || (y - dy < 2) || (dy + dh - 1 - y < 2);
                                            if (border) accessor.GetRowSpan(y)[x] = color;
                                        }
                                    }
                                });
                            }

                            var outPath = Path.Combine(saidaDir, $"detect_{idx:000}_ann_{Path.GetFileName(ann.ImagePath)}");
                            using (var fs = File.Create(outPath + ".png")) img.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                            var info = $"ann_box={gb.X},{gb.Y},{gb.Width},{gb.Height}\ndetection={bestX},{bestY},{bestW},{bestH}\nscore={bestScore:F6}\n";
                            File.WriteAllText(outPath + ".txt", info);
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"detect-test sample error: {ex.Message}");
                        }
                        idx++;
                    }
                    Console.WriteLine($"detect-test completed. Outputs in {saidaDir}");
                    return;
                }

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
                            // debug: optionally emit gradient magnitude before optimizer step
                            var dbg = Environment.GetEnvironmentVariable("DEBUG_GRADS");
                            if (!string.IsNullOrEmpty(dbg) && (dbg == "1" || dbg.Equals("true", StringComparison.OrdinalIgnoreCase)))
                            {
                                double gradSum = 0.0;
                                foreach (var pp in paramList)
                                {
                                    if (pp?.Grad != null)
                                    {
                                        var garr = pp.Grad;
                                        for (int gi = 0; gi < garr.Length; gi++) gradSum += Math.Abs(garr[gi]);
                                    }
                                }
                                Console.WriteLine($"GradSum before step: {gradSum:E6}");
                            }
                            // apply weight decay to gradients if requested (L2)
                            if (weightDecay > 0.0)
                            {
                                foreach (var pp in paramList)
                                {
                                    if (pp?.Grad == null) continue;
                                    var g = pp.Grad;
                                    for (int gi = 0; gi < g.Length; gi++) g[gi] += weightDecay * pp[gi];
                                }
                            }
                            // global grad clipping
                            if (gradClip > 0.0)
                            {
                                double sumsq = 0.0;
                                foreach (var pp in paramList)
                                {
                                    if (pp?.Grad == null) continue;
                                    var g = pp.Grad;
                                    for (int gi = 0; gi < g.Length; gi++) sumsq += g[gi] * g[gi];
                                }
                                var norm = Math.Sqrt(sumsq);
                                if (norm > 0 && norm > gradClip)
                                {
                                    var scale = gradClip / (norm + 1e-12);
                                    foreach (var pp in paramList)
                                    {
                                        if (pp?.Grad == null) continue;
                                        var g = pp.Grad;
                                        for (int gi = 0; gi < g.Length; gi++) g[gi] *= scale;
                                    }
                                }
                            }
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

                    // Scheduler step: reduce LR on plateau
                    if (scheduler != null && !double.IsNaN(avgLoss))
                    {
                        var reduced = scheduler.Step(avgLoss);
                        if (reduced && optimizer != null)
                        {
                            optimizer.Lr = scheduler.CurrentLr;
                            Console.WriteLine($"ReduceOnPlateau: reduced LR to {scheduler.CurrentLr:E6}");
                        }
                    }

                    // checkpoint save
                    try
                    {
                        // save model and optimizer state
                        var tmp = pesosDirRoot + ".tmp_" + DateTime.UtcNow.Ticks;
                        if (Directory.Exists(tmp)) Directory.Delete(tmp, true);
                        Directory.CreateDirectory(tmp);
                        model.SaveWeights(tmp);
                        optimizer?.SaveState(tmp);
                        var metaObj = new System.Collections.Generic.Dictionary<string, object>();
                        metaObj["epoch"] = epoch;
                        metaObj["lr"] = optimizer?.Lr ?? hp.InitialLearningRate;
                        metaObj["timestamp"] = DateTime.UtcNow;
                        if (scheduler != null)
                        {
                            var sched = new System.Collections.Generic.Dictionary<string, object>();
                            sched["bestLoss"] = scheduler.BestLoss;
                            sched["epochsSinceImprovement"] = scheduler.EpochsSinceImprovement;
                            sched["currentLr"] = scheduler.CurrentLr;
                            metaObj["scheduler"] = sched;
                        }
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

        // Simple Reduce-on-Plateau scheduler used by the runner.
        private class ReduceOnPlateauScheduler
        {
            public double BestLoss { get; set; } = double.PositiveInfinity;
            public int EpochsSinceImprovement { get; set; } = 0;
            public double CurrentLr { get; set; }
            public double MinLr { get; }
            public double Factor { get; }
            public int Patience { get; }

            public ReduceOnPlateauScheduler(double initialLr, double minLr, double factor, int patience)
            {
                CurrentLr = initialLr;
                MinLr = minLr;
                Factor = factor;
                Patience = Math.Max(1, patience);
            }

            // returns true if LR was reduced
            public bool Step(double loss)
            {
                if (double.IsNaN(loss)) return false;
                // consider improvement only if strictly less
                if (loss < BestLoss - 1e-12)
                {
                    BestLoss = loss;
                    EpochsSinceImprovement = 0;
                    return false;
                }
                EpochsSinceImprovement++;
                if (EpochsSinceImprovement >= Patience)
                {
                    var newLr = Math.Max(MinLr, CurrentLr * Factor);
                    if (newLr < CurrentLr - 1e-20)
                    {
                        CurrentLr = newLr;
                        EpochsSinceImprovement = 0;
                        return true;
                    }
                    EpochsSinceImprovement = 0;
                }
                return false;
            }
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
