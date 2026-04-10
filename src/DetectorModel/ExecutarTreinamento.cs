using System;
using System.IO;
using DetectorModel.dados;
using DetectorModel.modelo;
using Bionix.ML.nucleo.tensor;

namespace DetectorModel
{
    public class ExecutarTreinamento
    {
        public static void Main(string[] args)
        {
            // Paths (workspace-relative); adjust if necessary
            var datasetRoot = Path.Combine("DATASET", "dataset_deteccao");
            var annotationsFile = Path.Combine(datasetRoot, "wider_face_annotations", "wider_face_split", "wider_face_train_bbx_gt.txt");
            var imagesFolder = Path.Combine(datasetRoot, "WIDER_train", "WIDER_train", "images", "13--Interview");

            Console.WriteLine("Inicializando DataLoader...");
            var loader = new DataLoader(annotationsFile, imagesFolder);

            Console.WriteLine("Criando modelo RetinaFace (esqueleto)...");
            var model = new RetinaFaceModel();

            int batchSize = 4;
            Console.WriteLine($"Iterando batches (tamanho={batchSize})...");

            var ctx = new Bionix.ML.computacao.ComputacaoCPUContexto();
            var rnd = new Random();

            // initialize model weights
            model.InitializeWeights(ctx);

            int epochs = 3;
            int batchIndex = 0;

            var saidaDir = Path.Combine("SAIDA");
            Directory.CreateDirectory(saidaDir);
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
                                using var img = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgba32>(sample.ImagePath);
                                DrawBoxes(img, sample.Boxes, SixLabors.ImageSharp.Color.Green);
                                DrawBoxes(img, detections, SixLabors.ImageSharp.Color.Blue);
                                var outPathBase = Path.Combine(saidaDir, $"epoch_{epoch}_batch_{localBatch}_{Path.GetFileNameWithoutExtension(sample.ImagePath)}");
                                var outPath = Path.ChangeExtension(outPathBase, ".png");
                                using var fs = File.OpenWrite(outPath);
                                img.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Erro ao salvar imagem anotada: {ex.Message}");
                            }
                        }
                    }
                    localBatch++;
                }

                // Save weights at end of epoch
                var epochPesoDir = Path.Combine(pesosDirRoot, $"epoch_{epoch}");
                model.SaveWeights(epochPesoDir);
                Console.WriteLine($"Pesos salvos em {epochPesoDir}");
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
