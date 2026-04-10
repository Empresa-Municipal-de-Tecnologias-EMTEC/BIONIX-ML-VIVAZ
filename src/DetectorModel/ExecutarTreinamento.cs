using System;
using System.IO;
using DetectorModel.dados;
using DetectorModel.modelo;
using Bionix.ML.nucleo.tensor;
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
            var pesosDirRoot = Path.Combine("PESOS", "DETECTOR");
            Directory.CreateDirectory(pesosDirRoot);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                Console.WriteLine($"Época {epoch}");
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

                            // Build loss tensor: sumSquares(clsOut) + sumSquares(regOut)
                            var loss1 = clsOut.SumSquares();
                            var loss2 = regOut.SumSquares();
                            var totalLoss = loss1.Add(loss2);
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
                                    // Quick test: write original bytes to SAIDA to ensure a write actually occurs
                                    var outPathBase = Path.Combine(saidaDir, $"epoch_{epoch}_batch_{localBatch}_{Path.GetFileName(sample.ImagePath)}");
                                    var outPath = outPathBase; // preserve original extension
                                    Console.WriteLine($" Quick-copying original image to: {outPath}");
                                    try
                                    {
                                        var bytes = File.ReadAllBytes(sample.ImagePath);
                                        File.WriteAllBytes(outPath, bytes);
                                        File.AppendAllText(saidaLog, $"{DateTime.UtcNow:o} QUICK_COPY {sample.ImagePath} -> {outPath}\n");
                                    }
                                    catch (Exception ex)
                                    {
                                        Console.WriteLine($"Quick-copy failed: {ex.Message}");
                                        File.AppendAllText(saidaLog, $"{DateTime.UtcNow:o} QUICK_COPY_ERROR {ex.Message}\n");
                                    }
                                    Console.WriteLine($" Quick-copy exists: {File.Exists(outPath)}");
                                    Console.Out.Flush();
                                    Console.WriteLine("Quick test mode: copied image, exiting.");
                                    Console.Out.Flush();
                                    return;
                                }

                                using var img = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgba32>(sample.ImagePath);
                                DrawBoxes(img, sample.Boxes, SixLabors.ImageSharp.Color.Green);
                                DrawBoxes(img, detections, SixLabors.ImageSharp.Color.Blue);
                                var outPathBase = Path.Combine(saidaDir, $"epoch_{epoch}_batch_{localBatch}_{Path.GetFileNameWithoutExtension(sample.ImagePath)}");
                                var outPath = Path.ChangeExtension(outPathBase, ".png");
                                Console.WriteLine($" Saving annotated image to: {outPath}");
                                using var fs = File.OpenWrite(outPath);
                                img.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                                Console.WriteLine($" Saved exists: {File.Exists(outPath)}");
                                File.AppendAllText(saidaLog, $"{DateTime.UtcNow:o} ANNOTATED_SAVE {outPath} Exists={File.Exists(outPath)}\n");
                                Console.Out.Flush();
                                if (quickTest)
                                {
                                    Console.WriteLine("Quick test mode: annotated image saved, exiting.");
                                    Console.Out.Flush();
                                    return;
                                }
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
            }

            Console.WriteLine("Execução de treinamento (esqueleto) finalizada.");
        }

        private static void DrawBoxes(SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32> img, System.Collections.Generic.List<DetectorModel.dados.Box> boxes, SixLabors.ImageSharp.Color color)
        {
            int w = img.Width;
            int h = img.Height;
            foreach (var b in boxes)
            {
                int x0 = Math.Max(0, b.X);
                int y0 = Math.Max(0, b.Y);
                int x1 = Math.Min(w - 1, b.X + b.Width - 1);
                int y1 = Math.Min(h - 1, b.Y + b.Height - 1);
                // top & bottom
                for (int x = x0; x <= x1; x++)
                {
                    if (y0 >= 0 && y0 < h) img[x, y0] = color.ToPixel<SixLabors.ImageSharp.PixelFormats.Rgba32>();
                    if (y1 >= 0 && y1 < h) img[x, y1] = color.ToPixel<SixLabors.ImageSharp.PixelFormats.Rgba32>();
                }
                // left & right
                for (int y = y0; y <= y1; y++)
                {
                    if (x0 >= 0 && x0 < w) img[x0, y] = color.ToPixel<SixLabors.ImageSharp.PixelFormats.Rgba32>();
                    if (x1 >= 0 && x1 < w) img[x1, y] = color.ToPixel<SixLabors.ImageSharp.PixelFormats.Rgba32>();
                }
            }
        }
    }
}
