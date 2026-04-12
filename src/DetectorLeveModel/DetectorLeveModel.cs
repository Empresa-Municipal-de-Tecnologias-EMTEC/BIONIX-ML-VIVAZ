using System;
using System.IO;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.dados.serializacao;

namespace DetectorLeveModel
{
    public class DetectorLeve
    {
        // MLP parameters: input 400 -> hidden 64 -> output 1
        public Tensor W1 { get; private set; }
        public Tensor b1 { get; private set; }
        public Tensor W2 { get; private set; }
        public Tensor b2 { get; private set; }

        public DetectorLeve() { }

        public void InitializeWeights(ComputacaoContexto ctx)
        {
            var fabrica = new FabricaTensor(ctx ?? new ComputacaoCPUContexto());
            W1 = fabrica.Criar(400, 64);
            b1 = fabrica.Criar(1, 64);
            W2 = fabrica.Criar(64, 1);
            b2 = fabrica.Criar(1, 1);
            var rnd = new Random(1234);
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
            // flatten to [1,400]
            var x = fabrica.Criar(1, input.Size);
            for (int i = 0; i < input.Size; i++) x[i] = input[i];

            // hidden = sigmoid(x * W1 + b1)
            var hidden = x.MatMul(W1); // [1,64]
            // add bias
            for (int i = 0; i < 64; i++) hidden[i] += b1[i];
            // apply sigmoid elementwise (use CPU helper implementation via creating tensor)
            hidden = SigmoidTensor(hidden, ctx);

            // out = sigmoid(hidden * W2 + b2)
            var outt = hidden.MatMul(W2); // [1,1]
            outt[0] += b2[0];
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

        public void SaveWeights(string dir)
        {
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            try { SerializadorTensor.SaveBinary(Path.Combine(dir, "w1.bin"), W1); } catch { }
            try { SerializadorTensor.SaveBinary(Path.Combine(dir, "b1.bin"), b1); } catch { }
            try { SerializadorTensor.SaveBinary(Path.Combine(dir, "w2.bin"), W2); } catch { }
            try { SerializadorTensor.SaveBinary(Path.Combine(dir, "b2.bin"), b2); } catch { }
        }

        public void LoadWeights(string dir)
        {
            try
            {
                var p = Path.Combine(dir, "w1.bin"); if (File.Exists(p) && W1 != null) { var t = SerializadorTensor.LoadBinary(p); if (t != null && t.Size == W1.Size) for (int i=0;i<W1.Size;i++) W1[i]=t[i]; }
                p = Path.Combine(dir, "b1.bin"); if (File.Exists(p) && b1 != null) { var t = SerializadorTensor.LoadBinary(p); if (t != null && t.Size == b1.Size) for (int i=0;i<b1.Size;i++) b1[i]=t[i]; }
                p = Path.Combine(dir, "w2.bin"); if (File.Exists(p) && W2 != null) { var t = SerializadorTensor.LoadBinary(p); if (t != null && t.Size == W2.Size) for (int i=0;i<W2.Size;i++) W2[i]=t[i]; }
                p = Path.Combine(dir, "b2.bin"); if (File.Exists(p) && b2 != null) { var t = SerializadorTensor.LoadBinary(p); if (t != null && t.Size == b2.Size) for (int i=0;i<b2.Size;i++) b2[i]=t[i]; }
            }
            catch { }
        }

        public System.Collections.Generic.IEnumerable<(string name, Tensor tensor)> GetNamedParameters()
        {
            yield return ("w1", W1);
            yield return ("b1", b1);
            yield return ("w2", W2);
            yield return ("b2", b2);
        }
    }
}
