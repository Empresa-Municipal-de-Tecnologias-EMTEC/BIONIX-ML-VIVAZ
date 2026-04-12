using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;
using IdentificadorModel.modelo;
using Bionix.ML.dados.imagem;
using Bionix.ML.dados.serializacao;
using Bionix.ML.nucleo.funcoesPerda;
using Bionix.ML.nucleo.otimizadores;

namespace IdentificadorModel
{
    public class ExecutarTreinamento
    {
        public class HyperParameters
        {
            public int NumEpochs { get; set; } = 5;
            public int BatchSize { get; set; } = 16;
            public double InitialLearningRate { get; set; } = 1e-3;
            public double FinalLearningRate { get; set; } = 1e-3;
        }

        // Treinar accepts hyperparameters, args (optional) and a computation context provided by the caller
        public static void treinar(HyperParameters hp, string[] args, ComputacaoContexto ctx)
        {
            if (args == null || args.Length == 0) { Console.WriteLine("Identificador training: missing identities root arg"); return; }
            var identitiesRoot = args[0];
            if (!Directory.Exists(identitiesRoot)) { Console.WriteLine($"Folder not found: {identitiesRoot}"); return; }

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

            var fabrica = new FabricaTensor(ctx);
            var W = fabrica.Criar(embeddingSize, labels.Length);
            W.RequiresGrad = true;
            var rnd = new Random(123);
            for (int i = 0; i < W.Size; i++) W[i] = (rnd.NextDouble() - 0.5) * 0.01;

            var paramList = new List<Tensor>();
            foreach (var kv in model.GetNamedParameters()) if (kv.tensor != null) paramList.Add(kv.tensor);
            paramList.Add(W);

            var optimizer = FabricaOtimizadores.CriarStatefulSGD(paramList, ctx, lr: hp.InitialLearningRate, momentum: 0.9);
            var bceFn = FabricaFuncoesPerda.CriarBCE(ctx);

            // Helper: reshape 1D embedding to [1,embeddingSize]
            Tensor ReshapeTo2D(Tensor vec)
            {
                var t = fabrica.Criar(1, vec.Size);
                for (int i = 0; i < vec.Size; i++) t[i] = vec[i];
                return t;
            }

            // Local sigmoid helper (supports CPU and CPUSIMD tensors)
            Tensor Sigmoid(Tensor input)
            {
                if (input is Bionix.ML.nucleo.tensor.TensorCPU srcCpu)
                {
                    var outT = fabrica.Criar(srcCpu.Shape[0], srcCpu.Shape[1]);
                    for (int i = 0; i < srcCpu.Size; i++) outT[i] = 1.0 / (1.0 + Math.Exp(-srcCpu[i]));
                    outT.RequiresGrad = true;
                    outT.GradFn = new Bionix.ML.grafo.CPU.SigmoidFunction(srcCpu, outT as Bionix.ML.nucleo.tensor.TensorCPU);
                    return outT;
                }
                else if (input is Bionix.ML.nucleo.tensor.TensorCPUSIMD srcSimd)
                {
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
                else throw new NotSupportedException("Unsupported tensor type for Sigmoid helper.");
            }

            for (int ep = 0; ep < hp.NumEpochs; ep++)
            {
                Console.WriteLine($"Epoch {ep}/{hp.NumEpochs}");
                samples = samples.OrderBy(x => rnd.Next()).ToList();
                double epochLoss = 0.0; int cnt = 0;
                foreach (var s in samples)
                {
                    try
                    {
                        var bmp = ManipuladorDeImagem.carregarBmpDeJPEG(s.path);
                        var resized = ManipuladorDeImagem.redimensionar(bmp, 112, 112);
                        var tensor = ManipuladorDeImagem.transformarEmTensor(resized, ctx);

                        var emb = model.Forward(tensor, ctx);
                        var emb2d = ReshapeTo2D(emb);
                        var logits = emb2d.MatMul(W);
                        var probs = Sigmoid(logits);

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

                // checkpoint
                var pesosDir = Path.Combine(Directory.GetCurrentDirectory(), "PESOS", "IDENTIFICADOR");
                Directory.CreateDirectory(pesosDir);
                try { model.SaveWeights(pesosDir); } catch { }
                try { SerializadorTensor.SaveBinary(Path.Combine(pesosDir, "classifier_W.bin"), W); } catch { }
                try { hp.FinalLearningRate = optimizer?.Lr ?? hp.InitialLearningRate; } catch { }
                Console.WriteLine($"Checkpoint saved to {pesosDir}");
            }
        }
    }
}
