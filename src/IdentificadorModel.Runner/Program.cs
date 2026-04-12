using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.dados.imagem;
using IdentificadorModel.modelo;
using Bionix.ML.nucleo.funcoesPerda;
using Bionix.ML.dados.serializacao;
using Bionix.ML.nucleo.otimizadores;

namespace IdentificadorModel.Runner
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            if (args == null || args.Length == 0)
            {
                Console.WriteLine("Usage: train <identities_root_folder> [--epochs N] [--lr LR]");
                return;
            }
            var cmd = args[0].ToLowerInvariant();
            if (cmd == "train" && args.Length >= 2)
            {
                var folder = args[1];
                int epochs = 5;
                double lr = 1e-3;
                for (int i = 2; i < args.Length; i++)
                {
                    if (args[i] == "--epochs" && i + 1 < args.Length) { int.TryParse(args[i + 1], out epochs); i++; }
                    if (args[i] == "--lr" && i + 1 < args.Length) { double.TryParse(args[i + 1], out lr); i++; }
                }
                Train(folder, epochs, lr);
                return;
            }
            Console.WriteLine("Unknown command");
        }

        private static void Train(string identitiesRoot, int epochs, double lr)
        {
            if (!Directory.Exists(identitiesRoot)) { Console.WriteLine($"Folder not found: {identitiesRoot}"); return; }

            // use CPU context for compatibility with SigmoidFunction implementation
            var ctx = new ComputacaoCPUContexto();

            // discover identities
            var dirs = Directory.GetDirectories(identitiesRoot);
            var labels = dirs.Select(d => Path.GetFileName(d)).ToArray();
            var samples = new List<(string path, int label)>();
            for (int i = 0; i < dirs.Length; i++)
            {
                var images = Directory.GetFiles(dirs[i]).Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".png", StringComparison.OrdinalIgnoreCase)).ToArray();
                foreach (var im in images) samples.Add((im, i));
            }
            if (samples.Count == 0) { Console.WriteLine("No images found in identities root."); return; }

            Console.WriteLine($"Found {labels.Length} identities, {samples.Count} images.");

            int embeddingSize = 128;
            var model = new ArcFaceModel(embeddingSize, ctx);
            model.InitializeWeights(ctx);

            // classifier weights: shape [embeddingSize, numClasses]
            var fabrica = new FabricaTensor(ctx);
            var W = fabrica.Criar(embeddingSize, labels.Length);
            W.RequiresGrad = true;
            var rnd = new Random(123);
            for (int i = 0; i < W.Size; i++) W[i] = (rnd.NextDouble() - 0.5) * 0.01;

            // parameters: model params + W
            var paramList = new List<Tensor>();
            foreach (var kv in model.GetNamedParameters()) if (kv.tensor != null) paramList.Add(kv.tensor);
            paramList.Add(W);

            var optimizer = FabricaOtimizadores.CriarStatefulSGD(paramList, ctx, lr: lr, momentum: 0.9);
            var bceFn = FabricaFuncoesPerda.CriarBCE(ctx);

            for (int ep = 0; ep < epochs; ep++)
            {
                Console.WriteLine($"Epoch {ep}/{epochs}");
                // shuffle
                samples = samples.OrderBy(x => rnd.Next()).ToList();
                double epochLoss = 0.0; int cnt = 0;
                foreach (var s in samples)
                {
                    try
                    {
                        var bmp = ManipuladorDeImagem.carregarBmpDeJPEG(s.path);
                        var resized = ManipuladorDeImagem.redimensionar(bmp, 112, 112);
                        var tensor = ManipuladorDeImagem.transformarEmTensor(resized, ctx);

                        var emb = model.Forward(tensor, ctx); // 1D tensor length embeddingSize
                        var emb2d = ReshapeTo2D(emb, ctx); // [1, embeddingSize]

                        var logits = emb2d.MatMul(W); // [1, numClasses]

                            // apply sigmoid elementwise with autograd for CPU
                            var probs = Sigmoid(logits, ctx);

                        // target one-hot
                        var targetArr = new double[labels.Length];
                        targetArr[s.label] = 1.0;
                        var target = fabrica.FromArray(new int[] { targetArr.Length }, targetArr);

                        var loss = bceFn(probs, target);
                        epochLoss += loss[0]; cnt++;

                        loss.Backward();
                        optimizer.Step();
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Sample error {s.path}: {ex.Message}");
                    }
                }
                Console.WriteLine($"Epoch {ep} avg loss = {(cnt>0?epochLoss/cnt:double.NaN):F6}");

                // checkpoint: save model FC and classifier W
                var pesosDir = Path.Combine(Directory.GetCurrentDirectory(), "PESOS", "IDENTIFICADOR");
                Directory.CreateDirectory(pesosDir);
                try { model.SaveWeights(pesosDir); } catch { }
                try { SerializadorTensor.SaveBinary(Path.Combine(pesosDir, "classifier_W.bin"), W); } catch { }
                Console.WriteLine($"Checkpoint saved to {pesosDir}");
            }
        }

        private static Tensor ReshapeTo2D(Tensor vec, ComputacaoContexto ctx)
        {
            var fabrica = new FabricaTensor(ctx);
            var t = fabrica.Criar(1, vec.Size);
            for (int i = 0; i < vec.Size; i++) t[i] = vec[i];
            return t;
        }

        private static Tensor Sigmoid(Tensor input, ComputacaoContexto ctx)
        {
            // Support both CPU and CPUSIMD tensors
            if (input is Bionix.ML.nucleo.tensor.TensorCPU srcCpu)
            {
                var fabrica = new FabricaTensor(ctx);
                var outT = fabrica.Criar(srcCpu.Shape[0], srcCpu.Shape[1]);
                for (int i = 0; i < srcCpu.Size; i++) outT[i] = 1.0 / (1.0 + Math.Exp(-srcCpu[i]));
                outT.RequiresGrad = true;
                outT.GradFn = new Bionix.ML.grafo.CPU.SigmoidFunction(srcCpu, outT as Bionix.ML.nucleo.tensor.TensorCPU);
                return outT;
            }
            else if (input is Bionix.ML.nucleo.tensor.TensorCPUSIMD srcSimd)
            {
                // create factory with the tensor's context to ensure matching TensorCPUSIMD type
                var fabrica = new FabricaTensor(srcSimd.Context);
                var outT = fabrica.Criar(srcSimd.Shape[0], srcSimd.Shape[1]);
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
            else
            {
                throw new NotSupportedException("Unsupported tensor type for Sigmoid helper.");
            }
        }
    }
}
