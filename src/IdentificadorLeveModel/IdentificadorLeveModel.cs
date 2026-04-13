using System;
using System.IO;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.dados.imagem;
using Bionix.ML.dados.imagem.bmp;
using Bionix.ML.dados.serializacao;
using System.Collections.Generic;

namespace IdentificadorLeveModel
{
    // Lightweight embedding extractor: MLP over 112x112 grayscale input -> 128-d embedding
    public class IdentificadorLeve
    {
        public Tensor W1 { get; private set; }
        public Tensor b1 { get; private set; }
        public Tensor W2 { get; private set; }
        public Tensor b2 { get; private set; }

        public int EmbeddingSize { get; } = 128;
        public int InputSide { get; } = 112;

        public IdentificadorLeve() { }

        public void InitializeWeights(ComputacaoContexto ctx)
        {
            var fabrica = new FabricaTensor(ctx ?? new ComputacaoCPUContexto());
            int inputDim = InputSide * InputSide; // grayscale
            W1 = fabrica.Criar(inputDim, 512);
            b1 = fabrica.Criar(1, 512);
            W2 = fabrica.Criar(512, EmbeddingSize);
            b2 = fabrica.Criar(1, EmbeddingSize);
            W1.RequiresGrad = true; b1.RequiresGrad = true; W2.RequiresGrad = true; b2.RequiresGrad = true;
            var rnd = new Random(1234);
            for (int i = 0; i < W1.Size; i++) W1[i] = (rnd.NextDouble() - 0.5) * 0.01;
            for (int i = 0; i < b1.Size; i++) b1[i] = 0.0;
            for (int i = 0; i < W2.Size; i++) W2[i] = (rnd.NextDouble() - 0.5) * 0.01;
            for (int i = 0; i < b2.Size; i++) b2[i] = 0.0;
        }

        // Forward: input is tensor from ManipuladorDeImagem.TransformarCropParaTensorGrayscale(...,112)
        // Returns a 2D tensor [1, EmbeddingSize] (keeps autograd graph)
        public Tensor Forward(Tensor input, ComputacaoContexto ctx)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            var fabrica = new FabricaTensor(ctx ?? new ComputacaoCPUContexto());
            // flatten to [1, inputDim]
            var x = fabrica.Criar(1, input.Size);
            for (int i = 0; i < input.Size; i++) x[i] = input[i];

            // hidden = sigmoid(x * W1 + b1)
            var hidden = x.MatMul(W1); // [1,512]
            hidden = hidden.Add(b1);
            hidden = SigmoidTensor(hidden, ctx);

            // out = hidden * W2 + b2 -> [1, EmbeddingSize]
            var outt = hidden.MatMul(W2);
            outt = outt.Add(b2);

            // Note: we return outt as-is (not L2-normalized tensor) so training can flow through
            return outt;
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

        public IEnumerable<(string name, Tensor tensor)> GetNamedParameters()
        {
            yield return ("w1", W1);
            yield return ("b1", b1);
            yield return ("w2", W2);
            yield return ("b2", b2);
        }

        // Singleton helper
        private static IdentificadorLeve _instance;
        private static readonly object _instLock = new object();

        public static IdentificadorLeve GetInstance(ComputacaoContexto ctx = null, string pesosDir = null)
        {
            if (_instance != null) return _instance;
            lock (_instLock)
            {
                if (_instance == null)
                {
                    var m = new IdentificadorLeve();
                    m.InitializeWeights(ctx ?? new ComputacaoCPUContexto());
                    if (!string.IsNullOrEmpty(pesosDir))
                    {
                        try { m.LoadWeights(pesosDir); } catch { }
                    }
                    _instance = m;
                }
            }
            return _instance;
        }

        // Create embedding tensor from a BMP image. Performs square-crop (center crop by trimming
        // the longer side), resizes to `InputSide`, converts to grayscale tensor and runs forward.
        public Tensor EmbedFromBmp(BMP bmp, ComputacaoContexto ctx)
        {
            if (bmp == null) throw new ArgumentNullException(nameof(bmp));
            // make square by trimming longer side (center crop)
            int w = bmp.Width, h = bmp.Height;
            BMP square = bmp;
            if (w != h)
            {
                if (h > w)
                {
                    int trim = h - w;
                    int top = trim / 2;
                    square = ManipuladorDeImagem.cortar(bmp, 0, top, w, w);
                }
                else
                {
                    int trim = w - h;
                    int left = trim / 2;
                    square = ManipuladorDeImagem.cortar(bmp, left, 0, h, h);
                }
            }
            var resized = ManipuladorDeImagem.redimensionar(square, InputSide, InputSide);
            var t = ManipuladorDeImagem.TransformarCropParaTensorGrayscale(resized, InputSide, ctx ?? new ComputacaoCPUContexto());
            var emb = this.Forward(t, ctx ?? new ComputacaoCPUContexto());
            return emb;
        }

        // Convert embedding tensor to array (optionally L2-normalized)
        public double[] EmbeddingToArray(Tensor emb, bool l2Normalize = true)
        {
            if (emb == null) throw new ArgumentNullException(nameof(emb));
            var arr = new double[EmbeddingSize];
            for (int i = 0; i < EmbeddingSize; i++) arr[i] = emb[i];
            if (l2Normalize)
            {
                double ss = 0.0; for (int i = 0; i < EmbeddingSize; i++) ss += arr[i] * arr[i];
                double nrm = Math.Sqrt(Math.Max(1e-12, ss));
                for (int i = 0; i < EmbeddingSize; i++) arr[i] /= nrm;
            }
            return arr;
        }

        // Cosine similarity between two arrays (assumes already L2-normalized). Returns value in [-1,1].
        public static double CosineSimilarity(double[] a, double[] b)
        {
            if (a == null || b == null) throw new ArgumentNullException();
            if (a.Length != b.Length) throw new ArgumentException("vector length mismatch");
            double s = 0.0; for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
            return s;
        }

        // Return percent similarity in [0..100]. We clamp negative cosine to 0.
        public static double SimilarityPercent(double[] a, double[] b)
        {
            var cos = CosineSimilarity(a, b);
            var v = Math.Max(0.0, cos);
            return v * 100.0;
        }

        // Compare two embeddings (arrays) and return percent and boolean whether above threshold (0.7 default)
        public static (double percent, bool same) CompareEmbeddings(double[] a, double[] b, double threshold = 0.7)
        {
            var p = SimilarityPercent(a, b);
            return (p, p >= threshold * 100.0);
        }

        // Given a list of embeddings, find the most similar to `query`. Returns index and percent.
        public static (int index, double percent) FindMostSimilar(IList<double[]> list, double[] query)
        {
            if (list == null || list.Count == 0) return (-1, 0.0);
            int bestIdx = -1; double best = -1.0;
            for (int i = 0; i < list.Count; i++)
            {
                try
                {
                    var p = SimilarityPercent(list[i], query);
                    if (p > best) { best = p; bestIdx = i; }
                }
                catch { }
            }
            return (bestIdx, best);
        }
    }
}

