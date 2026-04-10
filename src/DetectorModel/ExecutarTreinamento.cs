using System;
using System.IO;
using DetectorModel.dados;
using DetectorModel.modelo;
using Bionix.ML.nucleo.tensor;
using DetectorModel.modelo;
using Bionix.ML.nucleo.funcoesPerda.Focal;
using Bionix.ML.nucleo.funcoesPerda.SmoothL1;
using System.Linq;

namespace DetectorModel
{
    public class ExecutarTreinamento
    {
        public static void Main(string[] args)
        {
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
            // Use the root images folder so the annotation paths (which include subfolders) resolve correctly
            var imagesFolder = Path.Combine(datasetRoot, "WIDER_train", "WIDER_train", "images");

            Console.WriteLine("Inicializando DataLoader...");
            var loader = new DataLoader(annotationsFile, imagesFolder);

            // Diagnostic: print first few annotations and whether referenced image files exist
            Console.WriteLine("Diagnostic: preview annotations (up to 3)");
            Console.WriteLine($"CWD={Directory.GetCurrentDirectory()}");
            Console.WriteLine($"Resolved dataset root: {datasetRoot}");
            Console.WriteLine($"Annotations file: {annotationsFile}");
            int di = 0;
            foreach (var ann in loader.ReadAnnotations())
            {
                Console.WriteLine($" Ann: {ann.ImagePath} -> Exists: {File.Exists(ann.ImagePath)} | Boxes={ann.Boxes.Count}");
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
                        using var fs = File.OpenWrite(outPath);
                        img.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
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
                        DrawBoxes(img, dets, bluePx, thickness: 3);
                        using var fs2 = File.OpenWrite(outPath2);
                        img.Save(fs2, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
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

            Console.WriteLine("Criando modelo RetinaFace (esqueleto)...");
            var model = new RetinaFaceModel();

            int batchSize = 4;
            Console.WriteLine($"Iterando batches (tamanho={batchSize})...");

            var ctx = new Bionix.ML.computacao.ComputacaoCPUContexto();
            var rnd = new Random();

            // initialize model weights
            model.InitializeWeights(ctx);

            int epochs = 3;
            // QUICK_TEST_SAVE: if set to "1", run only one epoch and exit after first annotated image is written
            var quickTest = Environment.GetEnvironmentVariable("QUICK_TEST_SAVE") == "1";
            if (quickTest) epochs = 1;
            int batchIndex = 0;

            var saidaDir = Path.Combine(Directory.GetCurrentDirectory(), "SAIDA");
            Directory.CreateDirectory(saidaDir);
            var saidaLog = Path.Combine(saidaDir, "saida_ops.log");
            var pesosDirRoot = Path.Combine(Directory.GetCurrentDirectory(), "PESOS", "DETECTOR");
            Directory.CreateDirectory(pesosDirRoot);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                Console.WriteLine($"Época {epoch}");
                int testCounter = 0;
                int localBatch = 0;
                foreach (var batch in loader.GetBatchesTensors(batchSize, ctx))
                {
                    Console.WriteLine($" Batch {localBatch} com {batch.Count} amostras");
                    foreach (var sample in batch)
                    {
                        Console.WriteLine(sample.ImagePath);
                        if (sample.Tensor != null)
                        {
                            // Run model forward (placeholder)
                            var (clsOut, regOut) = model.Forward(sample.Tensor, ctx);

                            // Build detection losses via anchor matching + focal + smooth-L1
                            var clsCpu = clsOut as TensorCPU ?? throw new Exception("Expected TensorCPU for clsOut");
                            var regCpu = regOut as TensorCPU ?? throw new Exception("Expected TensorCPU for regOut");

                            // derive feature map sizes from input tensor
                            var inShape = sample.Tensor.Shape; int inH = inShape[0]; int inW = inShape[1];
                            int fh3 = inH, fw3 = inW;
                            int fh4 = Math.Max(1, fh3 / 2), fw4 = Math.Max(1, fw3 / 2);
                            int fh5 = Math.Max(1, fh4 / 2), fw5 = Math.Max(1, fw4 / 2);

                            // generate anchors per scale (one anchor per location: ratios=[1], scales=[1])
                            var anchors3 = UtilitarioAncoras.GenerateAnchors(fh3, fw3, baseSize: 32, ratios: new double[]{1.0}, scales: new double[]{1.0}, stride: 1);
                            var anchors4 = UtilitarioAncoras.GenerateAnchors(fh4, fw4, baseSize: 64, ratios: new double[]{1.0}, scales: new double[]{1.0}, stride: 2);
                            var anchors5 = UtilitarioAncoras.GenerateAnchors(fh5, fw5, baseSize: 128, ratios: new double[]{1.0}, scales: new double[]{1.0}, stride: 4);
                            var allAnchors = new System.Collections.Generic.List<BoxF>();
                            allAnchors.AddRange(anchors3); allAnchors.AddRange(anchors4); allAnchors.AddRange(anchors5);

                            // convert GT boxes to BoxF
                            var gts = new System.Collections.Generic.List<BoxF>();
                            foreach (var b in sample.Boxes) gts.Add(new BoxF(b.X, b.Y, b.Width, b.Height));

                            // match anchors
                            UtilitarioAncoras.MatchAnchors(allAnchors, gts, posIou: 0.5, negIou: 0.4, out int[] labels, out int[] matched, out double[][] bboxTargets);

                            // collect active indices (exclude ignore = -1)
                            var activeIdx = new System.Collections.Generic.List<int>();
                            var positiveIdx = new System.Collections.Generic.List<int>();
                            for (int i = 0; i < labels.Length; i++)
                            {
                                if (labels[i] != -1) activeIdx.Add(i);
                                if (labels[i] == 1) positiveIdx.Add(i);
                            }

                            var fabrica = new Bionix.ML.nucleo.tensor.FabricaTensor(ctx);

                            // classification: logits for active indices and binary targets
                            double[] clsLogitsArr = new double[activeIdx.Count];
                            double[] clsTgtArr = new double[activeIdx.Count];
                            for (int k = 0; k < activeIdx.Count; k++)
                            {
                                int ai = activeIdx[k];
                                clsLogitsArr[k] = clsCpu[ai];
                                clsTgtArr[k] = labels[ai] == 1 ? 1.0 : 0.0;
                            }
                            var clsPred = fabrica.FromArray(new int[]{clsLogitsArr.Length}, clsLogitsArr);
                            var clsTgt = fabrica.FromArray(new int[]{clsTgtArr.Length}, clsTgtArr);

                            var focal = FocalLoss.Loss(ctx, clsPred, clsTgt, alpha: 0.25, gamma: 2.0);

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
                                    for (int k = 0; k < 4; k++) regPredArr[p*4 + k] = regCpu[ai*4 + k];
                                    var tgt = bboxTargets[ai];
                                    for (int k = 0; k < 4; k++) regTgtArr[p*4 + k] = tgt[k];
                                }
                                var regPred = fabrica.FromArray(new int[]{regPredArr.Length}, regPredArr);
                                var regTgt = fabrica.FromArray(new int[]{regTgtArr.Length}, regTgtArr);
                                smoothL1Loss = SmoothL1Loss.Loss(ctx, regPred, regTgt);
                            }

                            var totalLoss = focal.Add(smoothL1Loss);
                            Console.WriteLine($" Loss (tensor) = {totalLoss[0]:F6}");

                            // Backward through autograd
                            totalLoss.Backward();

                            // Collect parameters and do SGD step
                            var parameters = new System.Collections.Generic.List<Bionix.ML.nucleo.tensor.Tensor>();
                            // model-level weights
                            if (model.BackboneWeight != null) parameters.Add(model.BackboneWeight);
                            if (model.HeadClsWeight != null) parameters.Add(model.HeadClsWeight);
                            if (model.HeadRegWeight != null) parameters.Add(model.HeadRegWeight);
                            // conv layer weights
                            if (model.Stem != null && model.Stem.Weight != null) parameters.Add(model.Stem.Weight);
                            if (model.HeadCls != null && model.HeadCls.Weight != null) parameters.Add(model.HeadCls.Weight);
                            if (model.HeadReg != null && model.HeadReg.Weight != null) parameters.Add(model.HeadReg.Weight);

                            Bionix.ML.nucleo.otimizadores.SGD.Step(parameters, lr: 1e-3);

                            // Simulate detections: copy ground truth boxes with small jitter
                            var detections = new System.Collections.Generic.List<DetectorModel.dados.Box>();
                            foreach (var b in sample.Boxes)
                            {
                                int jx = rnd.Next(-5, 6);
                                int jy = rnd.Next(-5, 6);
                                var db = new DetectorModel.dados.Box(b.X + jx, b.Y + jy, Math.Max(1, b.Width + rnd.Next(-5, 6)), Math.Max(1, b.Height + rnd.Next(-5, 6)));
                                detections.Add(db);
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
                                            DrawBoxes(img2, sample.Boxes, greenPx, thickness:3);
                                            DrawBoxes(img2, detections, bluePx, thickness:3);
                                        using var fs2 = File.OpenWrite(outPathAnnot);
                                        img2.Save(fs2, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                                        // save coordinates txt
                                        try
                                        {
                                            using var tw = File.CreateText(outPathTxt);
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
                                DrawBoxes(img, sample.Boxes, greenPx2);
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

                // Save main weights
                if (model.BackboneWeight != null)
                {
                    var p1 = Path.Combine(pesosDirRoot, "backbone.bin");
                    Bionix.ML.dados.serializacao.SerializadorTensor.SaveBinary(p1, model.BackboneWeight);
                    // save gradient if present
                    if (model.BackboneWeight.Grad != null)
                    {
                        Bionix.ML.dados.serializacao.SerializadorTensor.SaveBinary(Path.Combine(pesosDirRoot, "backbone.grad.bin"), model.BackboneWeight.Shape, model.BackboneWeight.Grad);
                    }
                }
                if (model.HeadClsWeight != null)
                {
                    var p2 = Path.Combine(pesosDirRoot, "head_cls.bin");
                    Bionix.ML.dados.serializacao.SerializadorTensor.SaveBinary(p2, model.HeadClsWeight);
                    if (model.HeadClsWeight.Grad != null)
                    {
                        Bionix.ML.dados.serializacao.SerializadorTensor.SaveBinary(Path.Combine(pesosDirRoot, "head_cls.grad.bin"), model.HeadClsWeight.Shape, model.HeadClsWeight.Grad);
                    }
                }
                if (model.HeadRegWeight != null)
                {
                    var p3 = Path.Combine(pesosDirRoot, "head_reg.bin");
                    Bionix.ML.dados.serializacao.SerializadorTensor.SaveBinary(p3, model.HeadRegWeight);
                    if (model.HeadRegWeight.Grad != null)
                    {
                        Bionix.ML.dados.serializacao.SerializadorTensor.SaveBinary(Path.Combine(pesosDirRoot, "head_reg.grad.bin"), model.HeadRegWeight.Shape, model.HeadRegWeight.Grad);
                    }
                }

                // Save optimizer/metadata (learning rate and epoch)
                try
                {
                    var meta = System.Text.Json.JsonSerializer.Serialize(new { epoch = epoch, lr = 1e-3, timestamp = DateTime.UtcNow });
                    File.WriteAllText(Path.Combine(pesosDirRoot, "meta.json"), meta);
                }
                catch { }

                Console.WriteLine($"Pesos salvos em {pesosDirRoot}");
                try
                {
                    var files = Directory.GetFiles(pesosDirRoot);
                    Console.WriteLine("Arquivos em PESOS/DETECTOR:");
                    foreach (var f in files) Console.WriteLine($" - {f}");
                }
                catch { }
            }

            Console.WriteLine("Execução de treinamento (esqueleto) finalizada.");
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
    }
}
