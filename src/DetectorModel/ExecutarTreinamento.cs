using System;
using System.IO;
using DetectorModel.dados;
using DetectorModel.modelo;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;
using DetectorModel.modelo;
using Bionix.ML.nucleo.funcoesPerda.Focal;
using Bionix.ML.nucleo.funcoesPerda.SmoothL1;
using System.Linq;

namespace DetectorModel
{
    public class ExecutarTreinamento
    {
        // Internal container for hyperparameters and runtime settings
        public class HyperParameters
        {
            public int NumEpochs { get; set; } = 100;
            public int BatchSize { get; set; } = 4;
            public int DefaultAnchorBase { get; set; } = 32;
            public double[] AnchorRatios { get; set; } = new double[] { 1.0 };
            public double[] AnchorScales { get; set; } = new double[] { 1.0 };
            public double PosIou { get; set; } = 0.5;
            public double NegIou { get; set; } = 0.4;
            public double DetectionScoreThreshold { get; set; } = 0.6;
            public int MaxDetections { get; set; } = 100;
            public bool DrawOnlyModelOutputs { get; set; } = false;
            // Optional explicit strides per feature level (p3,p4,p5). If null, strides will be derived from model head shapes.
            public int[] Strides { get; set; } = null;
        }

        public static void Main(string[] args)
        {
            var hp = new HyperParameters();
            // Parameters are set in code as requested
            hp.NumEpochs = 100;
            hp.BatchSize = 4;
            hp.DefaultAnchorBase = 32;
            hp.AnchorRatios = new double[] { 0.8, 1.0, 1.2 };
            hp.AnchorScales = new double[] { 4.0, 8.0, 12.0 };
            hp.PosIou = 0.7;
            hp.NegIou = 0.4;
            hp.DetectionScoreThreshold = 0.6;
            hp.MaxDetections = 4;
            hp.DrawOnlyModelOutputs = false;
            // sensible default strides for p3,p4,p5 (can be overridden via env/args)
            hp.Strides = new int[] { 8, 16, 32 };

            ComputacaoContexto ctx = new ComputacaoCPUContexto();
            
            try
            {
                treinar(hp, args, ctx);
            }
            finally
            {
                if (ctx is IDisposable d) d.Dispose();
            }
        }

        public static void treinar(HyperParameters hp, string[] args, ComputacaoContexto ctx){
            // Resolve DATASET folder by walking up from current directory (so runner works from any CWD)
            string resolvedDatasetRoot = null;
            var initialCwd = Directory.GetCurrentDirectory();
            var cur = initialCwd;
            while (cur != null)
            {
                if (Directory.Exists(Path.Combine(cur, "DATASET"))) { resolvedDatasetRoot = Path.Combine(cur, "DATASET", "dataset_deteccao"); break; }
                var parent = Directory.GetParent(cur);
                cur = parent?.FullName;
            }
            if (resolvedDatasetRoot == null)
            {
                // Try repository-specific known locations under workspace root
                var alt1 = Path.Combine(initialCwd, "BIONIX-ML-VIVAZ", "DATASET", "dataset_deteccao");
                var alt2 = Path.Combine(initialCwd, "BIONIX-ML", "DATASET", "dataset_deteccao");
                if (Directory.Exists(alt1)) resolvedDatasetRoot = alt1;
                else if (Directory.Exists(alt2)) resolvedDatasetRoot = alt2;
                else
                {
                    // fallback to workspace-relative path (may fail later)
                    resolvedDatasetRoot = Path.Combine("DATASET", "dataset_deteccao");
                }
            }

            var datasetRoot = resolvedDatasetRoot;
            var annotationsFile = Path.Combine(datasetRoot, "wider_face_annotations", "wider_face_split", "wider_face_train_bbx_gt.txt");
            // Prefer new CelebA-style dataset if present (faces_com_landmarks)
            DataLoader loader = null;
            // Try several likely locations for the new CelebA-style dataset
            string celebaFolder = null;
            var cand1 = Path.Combine(initialCwd, "BIONIX-ML-VIVAZ", "DATASET", "faces_com_landmarks");
            var cand2 = Path.Combine(initialCwd, "BIONIX-ML", "DATASET", "faces_com_landmarks");
            var cand3 = Path.Combine(initialCwd, "DATASET", "faces_com_landmarks");
            var cand4 = Path.Combine(Path.GetDirectoryName(datasetRoot) ?? string.Empty, "faces_com_landmarks");
            if (Directory.Exists(cand1)) celebaFolder = cand1;
            else if (Directory.Exists(cand2)) celebaFolder = cand2;
            else if (Directory.Exists(cand3)) celebaFolder = cand3;
            else if (Directory.Exists(cand4)) celebaFolder = cand4;
            if (!string.IsNullOrEmpty(celebaFolder))
            {
                Console.WriteLine($"Detected CelebA dataset at: {celebaFolder}");
                var celebaImages = Path.Combine(celebaFolder, "img_align_celeba", "img_align_celeba");
                // discover landmarks file (accept common variants)
                var lmCandidates = System.IO.Directory.GetFiles(celebaFolder, "list_landmarks*.csv");
                var celebaLm = lmCandidates.FirstOrDefault();
                var celebaBbox = Path.Combine(celebaFolder, "list_bbox_celeba.csv");
                if (!string.IsNullOrEmpty(celebaLm) && File.Exists(celebaLm) && Directory.Exists(celebaImages))
                {
                    annotationsFile = celebaLm;
                    // If bbox file exists, DataLoader will pick it up from same folder
                    var imagesFolderCeleba = celebaImages;
                    Console.WriteLine($"Using CelebA annotations: {annotationsFile}");
                    Console.WriteLine($"Using CelebA images folder: {imagesFolderCeleba}");
                    Console.WriteLine("Inicializando DataLoader...");
                    loader = new DataLoader(annotationsFile, imagesFolderCeleba);
                    // replace existing loader variable in outer scope by shadowing
                    goto loader_initialized;
                }
            }

            // Use the root images folder so the annotation paths (which include subfolders) resolve correctly
            var imagesFolder = Path.Combine(datasetRoot, "WIDER_train", "WIDER_train", "images");

            Console.WriteLine("Inicializando DataLoader...");
            loader ??= new DataLoader(annotationsFile, imagesFolder);
        loader_initialized: ;

            // Diagnostic: print first few annotations and whether referenced image files exist
            Console.WriteLine("Diagnostic: preview annotations (up to 3)");
            Console.WriteLine($"CWD={Directory.GetCurrentDirectory()}");
            Console.WriteLine($"Resolved dataset root: {datasetRoot}");
            Console.WriteLine($"Annotations file: {annotationsFile}");
            int di = 0;
            foreach (var ann in loader.ReadAnnotations())
            {
                Console.WriteLine($" Ann: {ann.ImagePath} -> Exists: {File.Exists(ann.ImagePath)} | Boxes={ann.Boxes.Count} | BoxSource={ann.BoxSource}");
                di++; if (di >= 3) break;
            }

            // QUICK_ANNOTATE_ONLY: annotate the first available sample and exit (skip training)
            var quickAnnotOnly = Environment.GetEnvironmentVariable("QUICK_ANNOTATE_ONLY") == "1";
            // QUICK_ANNOTATE_BOTH: annotate first sample with GT (green) and simulated detections (blue)
            var quickAnnotBoth = Environment.GetEnvironmentVariable("QUICK_ANNOTATE_BOTH") == "1";
            if (quickAnnotOnly)
            {
                var first = loader.ReadAnnotations().FirstOrDefault();
                if (first != null && File.Exists(first.ImagePath))
                {
                    try
                    {
                        var outDir = Path.Combine(Directory.GetCurrentDirectory(), "SAIDA");
                        Directory.CreateDirectory(outDir);
                        var outPath = Path.Combine(outDir, Path.GetFileNameWithoutExtension(first.ImagePath) + "_qa.png");
                        using var img = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgba32>(first.ImagePath);
                        var greenPx = new SixLabors.ImageSharp.PixelFormats.Rgba32(0, 255, 0, 255);
                        DrawBoxes(img, first.Boxes, greenPx, thickness: 3);
                        DrawLandmarks(img, first.Landmarks);
                        // draw small marker indicating origin near GT
                        DrawBoxSourceMarker(img, first.BoxSource, first.Boxes.Count > 0 ? first.Boxes[0] : (DetectorModel.dados.Box?)null);
                        using var fs = File.OpenWrite(outPath);
                        img.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                        // write coordinates and BOX_SOURCE
                        try
                        {
                            var txtPath = Path.Combine(outDir, Path.GetFileNameWithoutExtension(first.ImagePath) + "_qa.txt");
                            using var tw = File.CreateText(txtPath);
                            tw.WriteLine($"BOX_SOURCE {first.BoxSource}");
                            foreach (var b in first.Boxes) tw.WriteLine($"GT {b.X} {b.Y} {b.Width} {b.Height}");
                        }
                        catch { }
                        Console.WriteLine($"QUICK_ANNOTATE_ONLY saved: {outPath} Exists={File.Exists(outPath)}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"QUICK_ANNOTATE_ONLY failed: {ex.Message}");
                    }
                }
                else Console.WriteLine("QUICK_ANNOTATE_ONLY: no annotation found or image missing.");
                return;
            }

            if (quickAnnotBoth)
            {
                var first2 = loader.ReadAnnotations().FirstOrDefault();
                if (first2 != null && File.Exists(first2.ImagePath))
                {
                    try
                    {
                        var outDir2 = Path.Combine(Directory.GetCurrentDirectory(), "SAIDA");
                        Directory.CreateDirectory(outDir2);
                        var outPath2 = Path.Combine(outDir2, Path.GetFileNameWithoutExtension(first2.ImagePath) + "_both.png");
                        using var img = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgba32>(first2.ImagePath);
                        var greenPx = new SixLabors.ImageSharp.PixelFormats.Rgba32(0, 255, 0, 255);
                        var bluePx = new SixLabors.ImageSharp.PixelFormats.Rgba32(0, 0, 255, 255);
                        // simulated detections: jitter the GT boxes
                        var rndSim = new Random(123);
                        var dets = new System.Collections.Generic.List<DetectorModel.dados.Box>();
                        foreach (var b in first2.Boxes)
                        {
                            int jx = rndSim.Next(-8, 9);
                            int jy = rndSim.Next(-8, 9);
                            dets.Add(new DetectorModel.dados.Box(b.X + jx, b.Y + jy, Math.Max(1, b.Width + rndSim.Next(-8, 9)), Math.Max(1, b.Height + rndSim.Next(-8, 9))));
                        }
                        DrawBoxes(img, first2.Boxes, greenPx, thickness: 3);
                        DrawLandmarks(img, first2.Landmarks);
                        DrawBoxes(img, dets, bluePx, thickness: 3);
                        // draw marker near GT and write txt info
                        DrawBoxSourceMarker(img, first2.BoxSource, first2.Boxes.Count > 0 ? first2.Boxes[0] : (DetectorModel.dados.Box?)null);
                        using var fs2 = File.OpenWrite(outPath2);
                        img.Save(fs2, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                        try
                        {
                            var txtPath2 = Path.Combine(outDir2, Path.GetFileNameWithoutExtension(first2.ImagePath) + "_both.txt");
                            using var tw2 = File.CreateText(txtPath2);
                            tw2.WriteLine($"BOX_SOURCE {first2.BoxSource}");
                            foreach (var b in first2.Boxes) tw2.WriteLine($"GT {b.X} {b.Y} {b.Width} {b.Height}");
                            foreach (var d in dets) tw2.WriteLine($"DET {d.X} {d.Y} {d.Width} {d.Height}");
                        }
                        catch { }
                        Console.WriteLine($"QUICK_ANNOTATE_BOTH saved: {outPath2} Exists={File.Exists(outPath2)}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"QUICK_ANNOTATE_BOTH failed: {ex.Message}");
                    }
                }
                else Console.WriteLine("QUICK_ANNOTATE_BOTH: no annotation found or image missing.");
                return;
            }

            Console.WriteLine("Criando modelo RetinaFace...");
            var model = new RetinaFaceModel();

            // resume support: if --resume flag or RESUME=1 env var, try to load last checkpoint
            var resume = args != null && args.Length > 0 && args.Contains("--resume") || Environment.GetEnvironmentVariable("RESUME") == "1";
            int resumeFromEpoch = -1;
            if (resume)
            {
                try
                {
                    Console.WriteLine("Resume requested: attempting to load checkpoint from PESOS/DETECTOR...");
                    // load meta.json
                    var metaPath = Path.Combine(Directory.GetCurrentDirectory(), "PESOS", "DETECTOR", "meta.json");
                    if (File.Exists(metaPath))
                    {
                        var meta = System.Text.Json.JsonSerializer.Deserialize<System.Collections.Generic.Dictionary<string, object>>(File.ReadAllText(metaPath));
                        if (meta != null && meta.ContainsKey("epoch"))
                        {
                            try { resumeFromEpoch = Convert.ToInt32(meta["epoch"]); } catch { resumeFromEpoch = -1; }
                            Console.WriteLine($"Loaded meta.json: epoch={resumeFromEpoch}");
                        }
                    }

                    // helper to load tensor into existing target
                    void TryLoadInto(string name, Bionix.ML.nucleo.tensor.Tensor target)
                    {
                        try
                        {
                            var p = Path.Combine(Directory.GetCurrentDirectory(), "PESOS", "DETECTOR", name);
                            if (File.Exists(p) && target != null)
                            {
                                var t = Bionix.ML.dados.serializacao.SerializadorTensor.LoadBinary(p);
                                if (t != null && t.Size == target.Size)
                                {
                                    var arr = t.ToArray();
                                    for (int i = 0; i < target.Size; i++) target[i] = arr[i];
                                }
                            }
                        }
                        catch { }
                    }

                    // load all known model weights (new central loader)
                    try { model.LoadWeights(Path.Combine(Directory.GetCurrentDirectory(), "PESOS", "DETECTOR"), loadGrad: false); } catch { }
                    Console.WriteLine("Checkpoint weights loaded (where available).");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Resume load failed: {ex.Message}");
                }
            }

            // Configuration (from HyperParameters)
            int NUM_EPOCHS = hp.NumEpochs;
            int BATCH_SIZE = hp.BatchSize;
            int DEFAULT_ANCHOR_BASE = hp.DefaultAnchorBase; // base anchor size multiplier (will be scaled by feature stride)
            double[] ANCHOR_RATIOS = hp.AnchorRatios;
            double[] ANCHOR_SCALES = hp.AnchorScales;
            double POS_IOU = hp.PosIou;
            double NEG_IOU = hp.NegIou;
            // Detection display / decoding config
            double DETECTION_SCORE_THRESHOLD = hp.DetectionScoreThreshold; // only consider anchors with score >= this for decoding
            int MAX_DETECTIONS = hp.MaxDetections; // limit anchors considered per image before NMS
            bool DRAW_ONLY_MODEL_OUTPUTS = hp.DrawOnlyModelOutputs;
            Console.WriteLine($"Iterando batches (tamanho={BATCH_SIZE})...");

            // RNG seed: use SEED env var if provided, otherwise generate and persist
            int rngSeed;
            var seedEnv = Environment.GetEnvironmentVariable("SEED");
            if (!string.IsNullOrEmpty(seedEnv) && int.TryParse(seedEnv, out var parsed)) rngSeed = parsed;
            else rngSeed = (new Random()).Next();
            var rnd = new Random(rngSeed);

            // initialize model weights
            model.InitializeWeights(ctx);
            int epochs = NUM_EPOCHS;
            // QUICK_TEST_SAVE: if set to "1", run only one epoch and exit after first annotated image is written
            var quickTest = Environment.GetEnvironmentVariable("QUICK_TEST_SAVE") == "1";
            if (quickTest) epochs = 1;
            int batchIndex = 0;

            var saidaDir = Path.Combine(Directory.GetCurrentDirectory(), "SAIDA");
            Directory.CreateDirectory(saidaDir);
            var saidaLog = Path.Combine(saidaDir, "saida_ops.log");
            var pesosDirRoot = Path.Combine(Directory.GetCurrentDirectory(), "PESOS", "DETECTOR");
            Directory.CreateDirectory(pesosDirRoot);

            // resume offset (number of samples already processed) when resuming
            int resumeOffsetSamples = 0;
            if (resume)
            {
                try
                {
                    var metaPath = Path.Combine(Directory.GetCurrentDirectory(), "PESOS", "DETECTOR", "meta.json");
                    if (File.Exists(metaPath))
                    {
                        var meta = System.Text.Json.JsonSerializer.Deserialize<System.Collections.Generic.Dictionary<string, object>>(File.ReadAllText(metaPath));
                        if (meta != null && meta.ContainsKey("processedSamples"))
                        {
                            try { resumeOffsetSamples = Convert.ToInt32(meta["processedSamples"]); } catch { resumeOffsetSamples = 0; }
                        }
                        if (meta != null && meta.ContainsKey("rngSeed"))
                        {
                            try { rngSeed = Convert.ToInt32(meta["rngSeed"]); rnd = new Random(rngSeed); } catch { }
                        }
                    }
                }
                catch { }
            }

            int processedSamples = 0;

            // ReduceLROnPlateau scheduler state
            double bestEpochLoss = double.PositiveInfinity;
            int lrPatience = 3; // epochs to wait before reducing
            int epochsSinceImprovement = 0;
            double lrFactor = 0.5; // multiply lr by this when plateau
            double minLr = 1e-6;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                Console.WriteLine($"Época {epoch}");
                int testCounter = 0;
                int localBatch = 0;
                double epochLossSum = 0.0;
                int epochLossCount = 0;
                // prepare optimizer once per epoch using model parameters (include all trainable tensors)
                Bionix.ML.nucleo.otimizadores.StatefulSGD optimizer = null;
                var epochParameters = new System.Collections.Generic.List<Bionix.ML.nucleo.tensor.Tensor>();
                // collect all named params from model
                try
                {
                    foreach (var kv in model.GetNamedParameters())
                    {
                        if (kv.tensor != null) epochParameters.Add(kv.tensor);
                    }
                }
                catch { }

                // pass resume offset only on first epoch when resuming
                int startSample = (resume && epoch == 0) ? resumeOffsetSamples : 0;
                foreach (var batch in loader.GetBatchesTensors(BATCH_SIZE, ctx, startSample))
                {
                    Console.WriteLine($" Batch {localBatch} com {batch.Count} amostras");
                    foreach (var sample in batch)
                    {
                        Console.WriteLine(sample.ImagePath + $" | BoxSource={sample.BoxSource} | Boxes={sample.Boxes?.Count ?? 0}");
                            if (sample.Tensor != null)
                            {
                            // Run model forward (placeholder)
                            var (clsOut, regOut, lmkOut, clsHeadShapes) = model.Forward(sample.Tensor, ctx);

                            // Build detection losses via anchor matching + focal + smooth-L1
                            var clsTensor = clsOut as Bionix.ML.nucleo.tensor.Tensor ?? throw new Exception("Expected Tensor for clsOut");
                            var regTensor = regOut as Bionix.ML.nucleo.tensor.Tensor ?? throw new Exception("Expected Tensor for regOut");

                            // derive feature map sizes from model head shapes (ensures anchors match model outputs)
                            int fh3 = Math.Max(1, clsHeadShapes[0][0]);
                            int fw3 = Math.Max(1, clsHeadShapes[0][1]);
                            int fh4 = Math.Max(1, clsHeadShapes[1][0]);
                            int fw4 = Math.Max(1, clsHeadShapes[1][1]);
                            int fh5 = Math.Max(1, clsHeadShapes[2][0]);
                            int fw5 = Math.Max(1, clsHeadShapes[2][1]);

                            // derive input shape to estimate strides
                            var inShape = sample.Tensor.Shape; int inH = inShape[0]; int inW = inShape[1];

                            int stride3, stride4, stride5;
                            // prefer explicit strides from hyperparameters when provided
                            if (hp.Strides != null && hp.Strides.Length >= 3)
                            {
                                stride3 = Math.Max(1, hp.Strides[0]);
                                stride4 = Math.Max(1, hp.Strides[1]);
                                stride5 = Math.Max(1, hp.Strides[2]);
                            }
                            else
                            {
                                // compute stride per scale as average downsample factor
                                stride3 = Math.Max(1, (int)Math.Round(((double)inH / fh3 + (double)inW / fw3) / 2.0));
                                stride4 = Math.Max(1, (int)Math.Round(((double)inH / fh4 + (double)inW / fw4) / 2.0));
                                stride5 = Math.Max(1, (int)Math.Round(((double)inH / fh5 + (double)inW / fw5) / 2.0));
                            }

                            // compute base sizes scaled by stride so anchors adapt to model receptive field
                            int base3 = Math.Max(1, DEFAULT_ANCHOR_BASE * stride3);
                            int base4 = Math.Max(1, DEFAULT_ANCHOR_BASE * stride4);
                            int base5 = Math.Max(1, DEFAULT_ANCHOR_BASE * stride5);

                            // generate anchors per scale using computed base and stride
                            var anchors3 = UtilitarioAncoras.GenerateAnchors(fh3, fw3, baseSize: base3, ratios: ANCHOR_RATIOS, scales: ANCHOR_SCALES, stride: stride3);
                            var anchors4 = UtilitarioAncoras.GenerateAnchors(fh4, fw4, baseSize: base4, ratios: ANCHOR_RATIOS, scales: ANCHOR_SCALES, stride: stride4);
                            var anchors5 = UtilitarioAncoras.GenerateAnchors(fh5, fw5, baseSize: base5, ratios: ANCHOR_RATIOS, scales: ANCHOR_SCALES, stride: stride5);
                            var allAnchors = new System.Collections.Generic.List<BoxF>();
                            allAnchors.AddRange(anchors3); allAnchors.AddRange(anchors4); allAnchors.AddRange(anchors5);

                            // convert GT boxes to BoxF
                            var gts = new System.Collections.Generic.List<BoxF>();
                            foreach (var b in sample.Boxes) gts.Add(new BoxF(b.X, b.Y, b.Width, b.Height));

                            // match anchors (also produce landmark targets if available)
                            var landmarksForGt = new System.Collections.Generic.List<double[]>();
                            if (sample.Landmarks != null && sample.Landmarks.Count > 0)
                            {
                                // assume ordering corresponds to boxes list where available
                                foreach (var lm in sample.Landmarks) landmarksForGt.Add(lm);
                            }
                            UtilitarioAncoras.MatchAnchorsWithLandmarks(allAnchors, gts, landmarksForGt, posIou: POS_IOU, negIou: NEG_IOU, out int[] labels, out int[] matched, out double[][] bboxTargets, out double[][] landmarkTargets);

                            // collect active indices (exclude ignore = -1)
                            var activeIdx = new System.Collections.Generic.List<int>();
                            var positiveIdx = new System.Collections.Generic.List<int>();
                            for (int i = 0; i < labels.Length; i++)
                            {
                                if (labels[i] != -1) activeIdx.Add(i);
                                if (labels[i] == 1) positiveIdx.Add(i);
                            }

                            // defensive checks: ensure tensor sizes match expected anchors count
                            int A = allAnchors.Count;
                            if (clsTensor.Size < A)
                            {
                                    // Detailed diagnostic to help find mismatch between model outputs and generated anchors
                                    var anchors3Count = anchors3.Count;
                                    var anchors4Count = anchors4.Count;
                                    var anchors5Count = anchors5.Count;
                                    Console.WriteLine($"Runner error: cls tensor size {clsTensor.Size} < anchors {A}. Skipping this sample.");
                                    Console.WriteLine($" Diagnostic: inH={inH} inW={inW} -> anchors per scale: p3={anchors3Count}, p4={anchors4Count}, p5={anchors5Count} (sum={anchors3Count+anchors4Count+anchors5Count}).");
                                    try
                                    {
                                        var lmkLocal = lmkOut as Bionix.ML.nucleo.tensor.Tensor;
                                        Console.WriteLine($" Tensor sizes: cls={clsTensor.Size}, reg={regTensor.Size}, lmk={(lmkLocal!=null?lmkLocal.Size:-1)}");
                                    }
                                    catch { }
                                continue; // skip this sample to avoid IndexOutOfRange
                            }
                            if (regTensor.Size < A * 4)
                            {
                                Console.WriteLine($"Runner error: reg tensor size {regTensor.Size} < anchors*4 {A * 4}. Skipping this sample.");
                                continue;
                            }

                            var fabrica = new Bionix.ML.nucleo.tensor.FabricaTensor(ctx);

                            // classification: logits for active indices and binary targets
                            double[] clsLogitsArr = new double[activeIdx.Count];
                            double[] clsTgtArr = new double[activeIdx.Count];
                            for (int k = 0; k < activeIdx.Count; k++)
                            {
                                int ai = activeIdx[k];
                                clsLogitsArr[k] = clsTensor[ai];
                                clsTgtArr[k] = labels[ai] == 1 ? 1.0 : 0.0;
                            }
                            var clsPred = fabrica.FromArray(new int[]{clsLogitsArr.Length}, clsLogitsArr);
                            var clsTgt = fabrica.FromArray(new int[]{clsTgtArr.Length}, clsTgtArr);

                            var focal = Focal.Loss(ctx, clsPred, clsTgt, alpha: 0.25, gamma: 2.0);

                            

                            // regression: only positives
                            Tensor smoothL1Loss;
                            if (positiveIdx.Count == 0)
                            {
                                var zero = fabrica.Criar(1);
                                zero[0] = 0.0;
                                smoothL1Loss = zero;
                            }
                            else
                            {
                                double[] regPredArr = new double[positiveIdx.Count * 4];
                                double[] regTgtArr = new double[positiveIdx.Count * 4];
                                for (int p = 0; p < positiveIdx.Count; p++)
                                {
                                    int ai = positiveIdx[p];
                                    for (int k = 0; k < 4; k++) regPredArr[p*4 + k] = regTensor[ai*4 + k];
                                    var tgt = bboxTargets[ai];
                                    for (int k = 0; k < 4; k++) regTgtArr[p*4 + k] = tgt[k];
                                }
                                var regPred = fabrica.FromArray(new int[]{regPredArr.Length}, regPredArr);
                                var regTgt = fabrica.FromArray(new int[]{regTgtArr.Length}, regTgtArr);
                                smoothL1Loss = SmoothL1.Loss(ctx, regPred, regTgt);
                            }

                            // landmarks loss (only positives)
                            Tensor lmkLoss;
                                var lmkCpu = lmkOut as Bionix.ML.nucleo.tensor.Tensor ?? throw new Exception("Expected Tensor for lmkOut");
                            // check landmark tensor size before indexing
                            if (lmkCpu.Size < allAnchors.Count * 10)
                            {
                                Console.WriteLine($"Runner error: landmark tensor size {lmkCpu.Size} < anchors*10 {allAnchors.Count * 10}. Landmarks will be skipped for this sample.");
                            }
                            if (positiveIdx.Count == 0)
                            {
                                var zero = fabrica.Criar(1);
                                zero[0] = 0.0;
                                lmkLoss = zero;
                            }
                            else
                            {
                                double[] lmkPredArr = new double[positiveIdx.Count * 10];
                                double[] lmkTgtArr = new double[positiveIdx.Count * 10];
                                for (int p = 0; p < positiveIdx.Count; p++)
                                {
                                    int ai = positiveIdx[p];
                                    for (int k = 0; k < 10; k++) lmkPredArr[p*10 + k] = lmkCpu[ai*10 + k];
                                    var tgt = landmarkTargets[ai];
                                    for (int k = 0; k < 10; k++) lmkTgtArr[p*10 + k] = tgt[k];
                                }
                                var lmkPred = fabrica.FromArray(new int[]{lmkPredArr.Length}, lmkPredArr);
                                var lmkTgt = fabrica.FromArray(new int[]{lmkTgtArr.Length}, lmkTgtArr);
                                lmkLoss = SmoothL1.Loss(ctx, lmkPred, lmkTgt);
                            }

                            var totalLoss = focal.Add(smoothL1Loss).Add(lmkLoss);
                            Console.WriteLine($" Loss (tensor) = {totalLoss[0]:F6}");
                            // accumulate epoch loss for scheduler
                            try { epochLossSum += totalLoss[0]; epochLossCount++; } catch { }

                            // Backward through autograd
                            totalLoss.Backward();

                            // initialize epoch optimizer lazily on first batch
                            if (optimizer == null)
                            {
                                optimizer = new Bionix.ML.nucleo.otimizadores.StatefulSGD(epochParameters, lr: 1e-3, momentum: 0.9);
                                try { optimizer.LoadState(pesosDirRoot); } catch { }
                            }
                            optimizer.Step();

                            // increment processed samples counter for resume tracking
                            processedSamples += batch.Count;

                            // Decode model predictions -> boxes and apply NMS
                            var detections = new System.Collections.Generic.List<DetectorModel.dados.Box>();
                            try
                            {
                                var boxesList = new System.Collections.Generic.List<DetectorModel.modelo.BoxF>();
                                var scores = new System.Collections.Generic.List<double>();
                                int anchorCount = allAnchors.Count;
                                // collect scored anchors above threshold
                                var scored = new System.Collections.Generic.List<(int idx, double score)>();
                                for (int iA = 0; iA < anchorCount; iA++)
                                {
                                    double score = Sigmoid(clsTensor[iA]);
                                    if (score < DETECTION_SCORE_THRESHOLD) continue;
                                    scored.Add((iA, score));
                                }
                                // sort by score desc and limit to MAX_DETECTIONS
                                scored.Sort((a,b)=> b.score.CompareTo(a.score));
                                if (scored.Count > MAX_DETECTIONS) scored = scored.GetRange(0, MAX_DETECTIONS);
                                // decode only the selected anchors
                                foreach (var s in scored)
                                {
                                    int iA = s.idx; double score = s.score;
                                    var delta = new double[4];
                                    for (int k = 0; k < 4; k++) delta[k] = regTensor[iA*4 + k];
                                    var decoded = UtilitarioAncoras.Decode(allAnchors[iA], delta);
                                    boxesList.Add(decoded);
                                    scores.Add(score);
                                }
                                // apply NMS
                                var keep = UtilitarioAncoras.NMS(boxesList, scores, 0.4);
                                foreach (var idx in keep)
                                {
                                    var bf = boxesList[idx];
                                    int x = (int)Math.Round(bf.X);
                                    int y = (int)Math.Round(bf.Y);
                                    int w = (int)Math.Max(1, Math.Round(bf.W));
                                    int h = (int)Math.Max(1, Math.Round(bf.H));
                                    detections.Add(new DetectorModel.dados.Box(x, y, w, h));
                                }
                            }
                            catch (Exception)
                            {
                                // fallback to empty detections on error
                                detections = new System.Collections.Generic.List<DetectorModel.dados.Box>();
                            }

                            // Load original image and draw boxes
                            try
                            {
                                if (quickTest)
                                {
                                    // Quick test: try full annotation first (ImageSharp). If it fails, fallback to raw copy.
                                    var outBaseQuick = Path.Combine(saidaDir, $"epoca_{epoch:00}_{testCounter:0000}");
                                    var outPathAnnot = Path.ChangeExtension(outBaseQuick, ".png");
                                    var outPathTxt = Path.ChangeExtension(outBaseQuick, ".txt");
                                    Console.WriteLine($" Quick-annotating image to: {outPathAnnot}");
                                    try
                                    {
                                            using var img2 = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgba32>(sample.ImagePath);
                                            var greenPx = new SixLabors.ImageSharp.PixelFormats.Rgba32(0, 255, 0, 255);
                                            var bluePx = new SixLabors.ImageSharp.PixelFormats.Rgba32(0, 0, 255, 255);
                                            if (!DRAW_ONLY_MODEL_OUTPUTS)
                                            {
                                                DrawBoxes(img2, sample.Boxes, greenPx, thickness:3);
                                                DrawLandmarks(img2, sample.Landmarks);
                                                DrawBoxSourceMarker(img2, sample.BoxSource, sample.Boxes != null && sample.Boxes.Count > 0 ? sample.Boxes[0] : (DetectorModel.dados.Box?)null);
                                            }
                                            DrawBoxes(img2, detections, bluePx, thickness:3);
                                        using var fs2 = File.OpenWrite(outPathAnnot);
                                        img2.Save(fs2, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                                        // save coordinates txt
                                        try
                                        {
                                            using var tw = File.CreateText(outPathTxt);
                                            tw.WriteLine($"BOX_SOURCE {sample.BoxSource}");
                                            foreach (var b in sample.Boxes) tw.WriteLine($"GT {b.X} {b.Y} {b.Width} {b.Height}");
                                            foreach (var d in detections) tw.WriteLine($"DET {d.X} {d.Y} {d.Width} {d.Height}");
                                        }
                                        catch { }
                                        File.AppendAllText(saidaLog, $"{DateTime.UtcNow:o} QUICK_ANNOTATE {sample.ImagePath} -> {outPathAnnot}\n");
                                        Console.WriteLine($" Quick-annotated exists: {File.Exists(outPathAnnot)}");
                                        Console.Out.Flush();
                                        Console.WriteLine("Quick test mode: annotated image saved, exiting.");
                                        Console.Out.Flush();
                                        testCounter++;
                                        return;
                                    }
                                    catch (Exception ex)
                                    {
                                        Console.WriteLine($"Quick-annotate failed, falling back to raw copy: {ex.Message}");
                                        File.AppendAllText(saidaLog, $"{DateTime.UtcNow:o} QUICK_ANNOTATE_ERROR {ex.Message}\n");
                                        // fallback raw copy
                                        var outPathCopy = Path.Combine(saidaDir, $"epoca_{epoch:00}_{testCounter:0000}{Path.GetExtension(sample.ImagePath)}");
                                        try
                                        {
                                            var bytes = File.ReadAllBytes(sample.ImagePath);
                                            File.WriteAllBytes(outPathCopy, bytes);
                                            // write GT coords txt
                                            try
                                            {
                                                var outTxt = Path.ChangeExtension(outPathCopy, ".txt");
                                                using var tw2 = File.CreateText(outTxt);
                                                foreach (var b in sample.Boxes) tw2.WriteLine($"GT {b.X} {b.Y} {b.Width} {b.Height}");
                                            }
                                            catch { }
                                            File.AppendAllText(saidaLog, $"{DateTime.UtcNow:o} QUICK_COPY {sample.ImagePath} -> {outPathCopy}\n");
                                        }
                                        catch (Exception ex2)
                                        {
                                            Console.WriteLine($"Quick-copy failed: {ex2.Message}");
                                            File.AppendAllText(saidaLog, $"{DateTime.UtcNow:o} QUICK_COPY_ERROR {ex2.Message}\n");
                                        }
                                        Console.WriteLine($" Quick-copy exists: {File.Exists(outPathCopy)}");
                                        Console.Out.Flush();
                                        Console.WriteLine("Quick test mode: copied image, exiting.");
                                        Console.Out.Flush();
                                        testCounter++;
                                        return;
                                    }
                                }

                                using var img = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgba32>(sample.ImagePath);
                                var greenPx2 = new SixLabors.ImageSharp.PixelFormats.Rgba32(0, 255, 0, 255);
                                var bluePx2 = new SixLabors.ImageSharp.PixelFormats.Rgba32(0, 0, 255, 255);
                                            if (!DRAW_ONLY_MODEL_OUTPUTS)
                                            {
                                                DrawBoxes(img, sample.Boxes, greenPx2);
                                                DrawLandmarks(img, sample.Landmarks);
                                                DrawBoxSourceMarker(img, sample.BoxSource, sample.Boxes != null && sample.Boxes.Count > 0 ? sample.Boxes[0] : (DetectorModel.dados.Box?)null);
                                            }
                                DrawBoxes(img, detections, bluePx2);
                                var outBase = Path.Combine(saidaDir, $"epoca_{epoch:00}_{testCounter:0000}");
                                var outPath = Path.ChangeExtension(outBase, ".png");
                                var outTxtPath = Path.ChangeExtension(outBase, ".txt");
                                Console.WriteLine($" Saving annotated image to: {outPath}");
                                using var fs = File.OpenWrite(outPath);
                                img.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                                // write coords file
                                try
                                {
                                    using var tw = File.CreateText(outTxtPath);
                                    tw.WriteLine($"BOX_SOURCE {sample.BoxSource}");
                                    foreach (var b in sample.Boxes) tw.WriteLine($"GT {b.X} {b.Y} {b.Width} {b.Height}");
                                    foreach (var d in detections) tw.WriteLine($"DET {d.X} {d.Y} {d.Width} {d.Height}");
                                }
                                catch { }
                                Console.WriteLine($" Saved exists: {File.Exists(outPath)}");
                                File.AppendAllText(saidaLog, $"{DateTime.UtcNow:o} ANNOTATED_SAVE {outPath} Exists={File.Exists(outPath)}\n");
                                Console.Out.Flush();
                                if (quickTest)
                                {
                                    Console.WriteLine("Quick test mode: annotated image saved, exiting.");
                                    Console.Out.Flush();
                                    testCounter++;
                                    return;
                                }
                                testCounter++;
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Erro ao salvar imagem anotada: {ex.Message}");
                                Console.WriteLine(ex.StackTrace);
                            }
                        }
                    }
                    localBatch++;
                }

                // Save weights (only keep last epoch): clear previous files and write current state
                // Ensure directory exists and is emptied
                if (!Directory.Exists(pesosDirRoot)) Directory.CreateDirectory(pesosDirRoot);
                foreach (var f in Directory.GetFiles(pesosDirRoot)) File.Delete(f);

                // Evaluate epoch loss and apply ReduceLROnPlateau if configured
                try
                {
                    double epochAvgLoss = double.NaN;
                    if (epochLossCount > 0) epochAvgLoss = epochLossSum / epochLossCount;
                    if (!double.IsNaN(epochAvgLoss))
                    {
                        Console.WriteLine($"Epoch {epoch} average loss = {epochAvgLoss:F6}");
                        if (epochAvgLoss < bestEpochLoss - 1e-12)
                        {
                            bestEpochLoss = epochAvgLoss;
                            epochsSinceImprovement = 0;
                        }
                        else
                        {
                            epochsSinceImprovement++;
                            if (epochsSinceImprovement >= lrPatience && optimizer != null)
                            {
                                var newLr = Math.Max(minLr, optimizer.Lr * lrFactor);
                                if (newLr < optimizer.Lr)
                                {
                                    optimizer.Lr = newLr;
                                    Console.WriteLine($"ReduceLROnPlateau: reducing lr to {optimizer.Lr:E6}");
                                }
                                epochsSinceImprovement = 0;
                            }
                        }
                    }
                    else
                    {
                        Console.WriteLine("Epoch average loss: no samples recorded.");
                    }

                    // Save checkpoint atomically: write to temp dir then swap
                    
                    var tmpDir = pesosDirRoot + ".tmp_" + DateTime.UtcNow.Ticks;
                    if (Directory.Exists(tmpDir)) Directory.Delete(tmpDir, true);
                    Directory.CreateDirectory(tmpDir);
                    // Save model files into temp dir
                    try { model.SaveWeights(tmpDir); } catch (Exception ex) { Console.WriteLine($"Error saving model weights to tmp: {ex.Message}"); }
                    // Save optimizer slots into temp dir
                    try { optimizer?.SaveState(tmpDir); } catch (Exception ex) { Console.WriteLine($"Error saving optimizer state to tmp: {ex.Message}"); }
                    // Write meta.json with rngSeed and processedSamples (persist current lr)
                    var metaObj = new { epoch = epoch, lr = (optimizer != null ? optimizer.Lr : 1e-3), timestamp = DateTime.UtcNow, rngSeed = rngSeed, processedSamples = processedSamples };
                    var metaJson = System.Text.Json.JsonSerializer.Serialize(metaObj);
                    File.WriteAllText(Path.Combine(tmpDir, "meta.json"), metaJson);

                    // Atomically replace existing dir
                    try
                    {
                        if (Directory.Exists(pesosDirRoot)) Directory.Delete(pesosDirRoot, true);
                        Directory.Move(tmpDir, pesosDirRoot);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Atomic swap failed, attempting fallback copy: {ex.Message}");
                        try
                        {
                            if (!Directory.Exists(pesosDirRoot)) Directory.CreateDirectory(pesosDirRoot);
                            foreach (var f in Directory.GetFiles(tmpDir)) File.Copy(f, Path.Combine(pesosDirRoot, Path.GetFileName(f)), overwrite: true);
                            Directory.Delete(tmpDir, true);
                        }
                        catch (Exception ex2) { Console.WriteLine($"Fallback copy failed: {ex2.Message}"); }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Checkpoint save failed: {ex.Message}");
                }

                Console.WriteLine($"Pesos salvos em {pesosDirRoot}");
                try
                {
                    var files = Directory.GetFiles(pesosDirRoot);
                    Console.WriteLine("Arquivos em PESOS/DETECTOR:");
                    foreach (var f in files) Console.WriteLine($" - {f}");
                }
                catch { }
            }

            Console.WriteLine("Execução de treinamento finalizada.");
        }

        private static void DrawBoxes(SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32> img, System.Collections.Generic.List<DetectorModel.dados.Box> boxes, SixLabors.ImageSharp.PixelFormats.Rgba32 px, int thickness = 2)
        {
            int w = img.Width;
            int h = img.Height;
            // pixel color provided by caller (opaque Rgba32)
            foreach (var b in boxes)
            {
                int x0 = Math.Max(0, b.X);
                int y0 = Math.Max(0, b.Y);
                int x1 = Math.Min(w - 1, b.X + b.Width - 1);
                int y1 = Math.Min(h - 1, b.Y + b.Height - 1);
                for (int t = 0; t < thickness; t++)
                {
                    int tx0 = x0 - t; int ty0 = y0 - t; int tx1 = x1 + t; int ty1 = y1 + t;
                    // clamp
                    tx0 = Math.Max(0, tx0); ty0 = Math.Max(0, ty0); tx1 = Math.Min(w - 1, tx1); ty1 = Math.Min(h - 1, ty1);
                    // horizontal
                    for (int x = tx0; x <= tx1; x++)
                    {
                        if (ty0 >= 0 && ty0 < h) img[x, ty0] = px;
                        if (ty1 >= 0 && ty1 < h) img[x, ty1] = px;
                    }
                    // vertical
                    for (int y = ty0; y <= ty1; y++)
                    {
                        if (tx0 >= 0 && tx0 < w) img[tx0, y] = px;
                        if (tx1 >= 0 && tx1 < w) img[tx1, y] = px;
                    }
                }
            }
        }

        // Draw a small filled marker indicating box source near provided box (tries top-right, then above, then below)
        private static void DrawBoxSourceMarker(SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32> img, string source, DetectorModel.dados.Box? nearBox = null)
        {
            int w = img.Width;
            int h = img.Height;
            // make marker smaller so it is less likely to overlap important facial regions
            int mw = Math.Min(48, Math.Max(16, w / 12));
            int mh = Math.Min(14, Math.Max(8, h / 60));
            SixLabors.ImageSharp.PixelFormats.Rgba32 px;
            if (string.Equals(source, "landmarks", StringComparison.OrdinalIgnoreCase)) px = new SixLabors.ImageSharp.PixelFormats.Rgba32(0, 200, 0, 255);
            else if (string.Equals(source, "bbox", StringComparison.OrdinalIgnoreCase)) px = new SixLabors.ImageSharp.PixelFormats.Rgba32(200, 0, 0, 255);
            else px = new SixLabors.ImageSharp.PixelFormats.Rgba32(200, 200, 0, 255);

            int startX = 0, startY = 0;
            if (nearBox.HasValue)
            {
                var nb = nearBox.Value;
                // Preferred: top-right corner of the box (outside the box)
                startX = nb.X + nb.Width - mw; // align right
                startY = nb.Y - mh - 2; // above the box

                // ensure within image bounds
                if (startX + mw >= w) startX = Math.Max(0, w - mw - 1);
                if (startX < 0) startX = Math.Max(0, nb.X);

                // If above the image, try placing below the box
                if (startY < 0)
                {
                    startY = nb.Y + nb.Height + 2;
                    if (startY + mh >= h) // if still out of bounds, fallback to inside top-left of box
                    {
                        startY = Math.Max(0, nb.Y);
                        startX = Math.Max(0, nb.X);
                    }
                }
            }

            for (int yy = 0; yy < mh; yy++)
            {
                for (int xx = 0; xx < mw; xx++)
                {
                    int pxX = startX + xx;
                    int pxY = startY + yy;
                    if (pxX >= 0 && pxX < w && pxY >= 0 && pxY < h) img[pxX, pxY] = px;
                }
            }
        }

        // Draw landmark points (small filled circles) on the image for visual debug
        private static void DrawLandmarks(SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32> img, System.Collections.Generic.List<double[]> landmarks)
        {
            if (landmarks == null) return;
            var px = new SixLabors.ImageSharp.PixelFormats.Rgba32(255, 0, 255, 255);
            int w = img.Width; int h = img.Height;
            foreach (var lm in landmarks)
            {
                if (lm == null || lm.Length < 10) continue;
                for (int i = 0; i < 5; i++)
                {
                    int lx = (int)Math.Round(lm[i*2 + 0]);
                    int ly = (int)Math.Round(lm[i*2 + 1]);
                    // draw small 3x3 square
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            int x = lx + dx; int y = ly + dy;
                            if (x >= 0 && x < w && y >= 0 && y < h) img[x, y] = px;
                        }
                    }
                }
            }
        }

        private static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
    }
}
