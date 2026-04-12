using System;
using System.IO;
using Bionix.ML.computacao;
using Bionix.ML.camadas;
using Bionix.ML.camadas.Interfaces;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.dados.serializacao;

namespace IdentificadorModel.modelo
{
    // Lightweight ArcFace-style embedding model (small backbone + FC embedding)
    public class ArcFaceModel
    {
        public IConvLayer Stem { get; private set; }
        public IResidualBlock[] Stage2 { get; private set; }
        public IResidualBlock[] Stage3 { get; private set; }
        public IResidualBlock[] Stage4 { get; private set; }

        // final fully-connected weights (shape: inChannels x EmbeddingSize)
        public Tensor FcWeight { get; private set; }

        public int EmbeddingSize { get; private set; }

        public ArcFaceModel(int embeddingSize = 128, ComputacaoContexto ctx = null)
        {
            EmbeddingSize = embeddingSize;
            Stem = FabricaCamadas.CriarConvLayer(inChannels: 3, outChannels: 32, kernelSize: 3, ctx: ctx);
            Stage2 = new IResidualBlock[] { FabricaCamadas.CriarResidualBlock(32, ctx), FabricaCamadas.CriarResidualBlock(32, ctx) };
            Stage3 = new IResidualBlock[] { FabricaCamadas.CriarResidualBlock(32, ctx), FabricaCamadas.CriarResidualBlock(32, ctx) };
            Stage4 = new IResidualBlock[] { FabricaCamadas.CriarResidualBlock(32, ctx) };
        }

        public void InitializeWeights(ComputacaoContexto ctx)
        {
            Stem.Initialize(ctx);
            foreach (var b in Stage2) b.Initialize(ctx);
            foreach (var b in Stage3) b.Initialize(ctx);
            foreach (var b in Stage4) b.Initialize(ctx);

            // allocate fc weights: inChannels = last stage channels (32)
            var fabrica = new FabricaTensor(ctx ?? new ComputacaoCPUContexto());
            FcWeight = fabrica.Criar(32, EmbeddingSize);
            // small random init
            var rnd = new Random(1234);
            for (int i = 0; i < FcWeight.Size; i++) FcWeight[i] = (rnd.NextDouble() - 0.5) * 0.01;
        }

        // Forward pass: returns normalized embedding (double[] via Tensor)
        public Tensor Forward(Tensor input, ComputacaoContexto ctx)
        {
            // backbone
            var x = Stem.Forward(input, ctx);
            foreach (var b in Stage2) x = b.Forward(x, ctx);
            x = DownsampleBy2(x, ctx);
            foreach (var b in Stage3) x = b.Forward(x, ctx);
            x = DownsampleBy2(x, ctx);
            foreach (var b in Stage4) x = b.Forward(x, ctx);

            // global average pooling over spatial dims -> vector length = channels
            var vec = GlobalAveragePool(x, ctx); // 1D tensor length = channels

            // ensure fc weights shape matches
            if (FcWeight == null || FcWeight.Size != 32 * EmbeddingSize)
            {
                var fabrica = new FabricaTensor(ctx ?? new ComputacaoCPUContexto());
                FcWeight = fabrica.Criar(32, EmbeddingSize);
                for (int i = 0; i < FcWeight.Size; i++) FcWeight[i] = 0.0;
            }

            // matmul: (1 x in) * (in x emb) => (1 x emb)
            var vec2d = ReshapeTo2D(vec, ctx); // shape [1, in]
            var fcOut = vec2d.MatMul(FcWeight); // [1, emb]

            // convert to 1D embedding and normalize (L2)
            var arr = fcOut.ToArray();
            // fcOut shape is [1,emb] so ToArray returns flattened length EmbeddingSize
            double sumsq = 0.0; for (int i = 0; i < arr.Length; i++) sumsq += arr[i] * arr[i];
            double norm = Math.Sqrt(Math.Max(1e-12, sumsq));
            var fabrica2 = new FabricaTensor(ctx ?? new ComputacaoCPUContexto());
            var emb = fabrica2.Criar(EmbeddingSize);
            for (int i = 0; i < EmbeddingSize; i++) emb[i] = arr[i] / norm;
            return emb;
        }

        private Tensor ReshapeTo2D(Tensor vec, ComputacaoContexto ctx)
        {
            // vec is 1D shape [C] -> produce 2D [1, C]
            var fabrica = new FabricaTensor(ctx ?? new ComputacaoCPUContexto());
            var t = fabrica.Criar(1, vec.Size);
            for (int i = 0; i < vec.Size; i++) t[i] = vec[i];
            return t;
        }

        private Tensor GlobalAveragePool(Tensor src, ComputacaoContexto ctx)
        {
            var s = src.Shape; // expected [h,w,c]
            int h = s.Length > 0 ? s[0] : 1;
            int w = s.Length > 1 ? s[1] : 1;
            int c = s.Length > 2 ? s[2] : 1;
            var fabrica = new FabricaTensor(ctx ?? new ComputacaoCPUContexto());
            var outT = fabrica.Criar(c);
            for (int ch = 0; ch < c; ch++)
            {
                double acc = 0.0;
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        int idx = (y * w + x) * c + ch;
                        acc += src[idx];
                    }
                outT[ch] = acc / (h * w);
            }
            outT.RequiresGrad = true;
            return outT;
        }

        private Tensor DownsampleBy2(Tensor src, ComputacaoContexto ctx)
        {
            var s = src.Shape; int h = s[0], w = s[1], c = s[2];
            var fabrica = new FabricaTensor(ctx ?? new ComputacaoCPUContexto());
            var dst = fabrica.Criar(Math.Max(1, h/2), Math.Max(1, w/2), c);
            for (int y = 0; y < dst.Shape[0]; y++)
            for (int x = 0; x < dst.Shape[1]; x++)
            for (int ch = 0; ch < c; ch++)
            {
                int sy = Math.Min(h-1, y*2);
                int sx = Math.Min(w-1, x*2);
                dst[(y * dst.Shape[1] + x) * c + ch] = src[(sy * w + sx) * c + ch];
            }
            dst.RequiresGrad = true;
            return dst;
        }

        public void SaveWeights(string dir)
        {
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            if (FcWeight != null)
            {
                var p = Path.Combine(dir, "fc_weight.bin");
                SerializadorTensor.SaveBinary(p, FcWeight);
            }
        }

        public void LoadWeights(string dir)
        {
            try
            {
                var p = Path.Combine(dir, "fc_weight.bin");
                if (File.Exists(p) && FcWeight != null)
                {
                    var loaded = SerializadorTensor.LoadBinary(p);
                    if (loaded != null && loaded.Size == FcWeight.Size)
                    {
                        for (int i = 0; i < FcWeight.Size; i++) FcWeight[i] = loaded[i];
                    }
                }
            }
            catch { }
        }

        // Expose named parameters for training + saving
        public System.Collections.Generic.IEnumerable<(string name, Tensor tensor)> GetNamedParameters()
        {
            // stem
            yield return ("stem", TryGetPropertyTensor(Stem, "Weight"));
            yield return ("stem.bias", TryGetPropertyTensor(Stem, "Bias"));
            int i = 0;
            foreach (var b in Stage2)
            {
                yield return ($"stage2_{i}_conv1", TryGetNestedTensor(b, "Conv1", "Weight"));
                yield return ($"stage2_{i}_conv1.bias", TryGetNestedTensor(b, "Conv1", "Bias"));
                yield return ($"stage2_{i}_conv2", TryGetNestedTensor(b, "Conv2", "Weight"));
                yield return ($"stage2_{i}_conv2.bias", TryGetNestedTensor(b, "Conv2", "Bias"));
                i++;
            }
            i = 0;
            foreach (var b in Stage3)
            {
                yield return ($"stage3_{i}_conv1", TryGetNestedTensor(b, "Conv1", "Weight"));
                yield return ($"stage3_{i}_conv1.bias", TryGetNestedTensor(b, "Conv1", "Bias"));
                yield return ($"stage3_{i}_conv2", TryGetNestedTensor(b, "Conv2", "Weight"));
                yield return ($"stage3_{i}_conv2.bias", TryGetNestedTensor(b, "Conv2", "Bias"));
                i++;
            }
            i = 0;
            foreach (var b in Stage4)
            {
                yield return ($"stage4_{i}_conv1", TryGetNestedTensor(b, "Conv1", "Weight"));
                yield return ($"stage4_{i}_conv1.bias", TryGetNestedTensor(b, "Conv1", "Bias"));
                yield return ($"stage4_{i}_conv2", TryGetNestedTensor(b, "Conv2", "Weight"));
                yield return ($"stage4_{i}_conv2.bias", TryGetNestedTensor(b, "Conv2", "Bias"));
                i++;
            }
            // fc
            if (FcWeight != null) yield return ("fc_weight", FcWeight);
        }

        private static Tensor TryGetPropertyTensor(object obj, string propName)
        {
            if (obj == null) return null;
            try
            {
                var t = obj.GetType();
                var pi = t.GetProperty(propName);
                if (pi != null) return pi.GetValue(obj) as Tensor;
                var fieldNames = new string[] { propName, "_" + propName.ToLower(), "_" + propName, propName.ToLower(), "m_" + propName.ToLower() };
                foreach (var fn in fieldNames)
                {
                    var fi = t.GetField(fn, System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public);
                    if (fi != null)
                    {
                        return fi.GetValue(obj) as Tensor;
                    }
                }
                return null;
            }
            catch { return null; }
        }

        private static Tensor TryGetNestedTensor(object obj, string childProp, string nestedProp)
        {
            if (obj == null) return null;
            try
            {
                var t = obj.GetType();
                var pi = t.GetProperty(childProp);
                object child = null;
                if (pi != null) child = pi.GetValue(obj);
                else
                {
                    var fieldNames = new string[] { childProp, "_" + childProp.ToLower(), "_" + childProp };
                    foreach (var fn in fieldNames)
                    {
                        var fi = t.GetField(fn, System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public);
                        if (fi != null) { child = fi.GetValue(obj); break; }
                    }
                }
                if (child == null) return null;
                var ct = child.GetType();
                var pj = ct.GetProperty(nestedProp);
                if (pj != null) return pj.GetValue(child) as Tensor;
                var childFieldNames = new string[] { nestedProp, "_" + nestedProp.ToLower(), "_" + nestedProp };
                foreach (var fn in childFieldNames)
                {
                    var fi = ct.GetField(fn, System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public);
                    if (fi != null) return fi.GetValue(child) as Tensor;
                }
                return null;
            }
            catch { return null; }
        }
    }
}
