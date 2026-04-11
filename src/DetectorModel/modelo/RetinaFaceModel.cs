using Bionix.ML.nucleo.tensor;
using Bionix.ML.computacao;
using Bionix.ML.dados;
using Bionix.ML.camadas;
using Bionix.ML.camadas.Interfaces;
using Bionix.ML.grafo;

namespace DetectorModel.modelo
{
    // RetinaFace-like model implemented with simple ResNet-style blocks, FPN and per-scale heads
    public class RetinaFaceModel
    {
        // backbone
        public IConvLayer Stem { get; private set; }
        public IResidualBlock[] Stage2 { get; private set; }
        public IResidualBlock[] Stage3 { get; private set; }
        public IResidualBlock[] Stage4 { get; private set; }

        // FPN
        public IFPN Pyramid { get; private set; }

        // per-scale heads
        public IDetectionHead HeadP3 { get; private set; }
        public IDetectionHead HeadP4 { get; private set; }
        public IDetectionHead HeadP5 { get; private set; }

        // Compatibility properties for existing training code (use reflection to access concrete fields)
        public Tensor BackboneWeight => TryGetPropertyTensor(Stem, "Weight");
        public Tensor HeadClsWeight => TryGetNestedTensor(HeadP3, "ClsConv", "Weight");
        public Tensor HeadRegWeight => TryGetNestedTensor(HeadP3, "RegConv", "Weight");
        public Tensor HeadLmkWeight => TryGetNestedTensor(HeadP3, "LandmarksConv", "Weight");
        public IConvLayer HeadCls => TryGetNestedLayer<IConvLayer>(HeadP3, "ClsConv");
        public IConvLayer HeadReg => TryGetNestedLayer<IConvLayer>(HeadP3, "RegConv");

        public RetinaFaceModel(ComputacaoContexto ctx = null)
        {
            Stem = FabricaCamadas.CriarConvLayer(inChannels: 3, outChannels: 32, kernelSize: 3, ctx: ctx);
            // small residual stages
            Stage2 = new IResidualBlock[] { FabricaCamadas.CriarResidualBlock(32, ctx), FabricaCamadas.CriarResidualBlock(32, ctx) };
            Stage3 = new IResidualBlock[] { FabricaCamadas.CriarResidualBlock(32, ctx), FabricaCamadas.CriarResidualBlock(32, ctx) };
            Stage4 = new IResidualBlock[] { FabricaCamadas.CriarResidualBlock(32, ctx) };

            Pyramid = FabricaCamadas.CriarFPN(inChannelsC5: 32, inChannelsC4: 32, inChannelsC3: 32, outChannels: 32, ctx: ctx);

            HeadP3 = FabricaCamadas.CriarDetectionHead(inChannels: 32, interChannels: 32, ctx: ctx);
            HeadP4 = FabricaCamadas.CriarDetectionHead(inChannels: 32, interChannels: 32, ctx: ctx);
            HeadP5 = FabricaCamadas.CriarDetectionHead(inChannels: 32, interChannels: 32, ctx: ctx);
        }

        public void InitializeWeights(ComputacaoContexto ctx)
        {
            Stem.Initialize(ctx);
            foreach (var b in Stage2) b.Initialize(ctx);
            foreach (var b in Stage3) b.Initialize(ctx);
            foreach (var b in Stage4) b.Initialize(ctx);
            Pyramid.Initialize(ctx);
            HeadP3.Initialize(ctx);
            HeadP4.Initialize(ctx);
            HeadP5.Initialize(ctx);
        }

        private static Tensor TryGetPropertyTensor(object obj, string propName)
        {
            if (obj == null) return null;
            try
            {
                var pi = obj.GetType().GetProperty(propName);
                if (pi == null) return null;
                return pi.GetValue(obj) as Tensor;
            }
            catch { return null; }
        }

        private static Tensor TryGetNestedTensor(object obj, string childProp, string nestedProp)
        {
            if (obj == null) return null;
            try
            {
                var pi = obj.GetType().GetProperty(childProp);
                if (pi == null) return null;
                var child = pi.GetValue(obj);
                if (child == null) return null;
                var pj = child.GetType().GetProperty(nestedProp);
                if (pj == null) return null;
                return pj.GetValue(child) as Tensor;
            }
            catch { return null; }
        }

        private static T TryGetNestedLayer<T>(object obj, string childProp) where T : class
        {
            if (obj == null) return null;
            try
            {
                var pi = obj.GetType().GetProperty(childProp);
                if (pi == null) return null;
                return pi.GetValue(obj) as T;
            }
            catch { return null; }
        }

        // Create a simple concatenated output: flatten per-scale cls, reg and landmarks into single tensors
        public (Tensor cls, Tensor reg, Tensor lmk, int[][] clsHeadShapes) Forward(Tensor input, ComputacaoContexto ctx)
        {
            // Stem
            var x = Stem.Forward(input, ctx);
            // Stage2
            foreach (var b in Stage2) x = b.Forward(x, ctx);
            var c3 = x; // treat as C3
            // downsample by 2 (naive conv stride simulated by taking every other pixel)
            var c4 = DownsampleBy2(c3, ctx);
            foreach (var b in Stage3) c4 = b.Forward(c4, ctx);
            var c5 = DownsampleBy2(c4, ctx);
            foreach (var b in Stage4) c5 = b.Forward(c5, ctx);

            var (p3, p4, p5) = Pyramid.Forward(c3, c4, c5, ctx);

            var (cls3, reg3, lmk3) = HeadP3.Forward(p3, ctx);
            var (cls4, reg4, lmk4) = HeadP4.Forward(p4, ctx);
            var (cls5, reg5, lmk5) = HeadP5.Forward(p5, ctx);

            // Optional debug: print per-scale tensor shapes and sizes when requested
            try
            {
                if (string.Equals(Environment.GetEnvironmentVariable("DEBUG_PER_SCALE_SHAPES"), "1", StringComparison.OrdinalIgnoreCase))
                {
                    string fmt(int[] s) => s == null ? "null" : "[" + string.Join(',', s) + "]";
                    System.Console.WriteLine("DEBUG_MODEL_SCALES: p3=" + fmt(p3.Shape) + " size=" + p3.Size +
                                             ", p4=" + fmt(p4.Shape) + " size=" + p4.Size +
                                             ", p5=" + fmt(p5.Shape) + " size=" + p5.Size);

                    System.Console.WriteLine("DEBUG_HEAD_CLS: cls3=" + fmt(cls3.Shape) + " size=" + cls3.Size +
                                             ", cls4=" + fmt(cls4.Shape) + " size=" + cls4.Size +
                                             ", cls5=" + fmt(cls5.Shape) + " size=" + cls5.Size);

                    System.Console.WriteLine("DEBUG_HEAD_REG: reg3=" + fmt(reg3.Shape) + " size=" + reg3.Size +
                                             ", reg4=" + fmt(reg4.Shape) + " size=" + reg4.Size +
                                             ", reg5=" + fmt(reg5.Shape) + " size=" + reg5.Size);

                    System.Console.WriteLine("DEBUG_HEAD_LMK: lmk3=" + fmt(lmk3.Shape) + " size=" + lmk3.Size +
                                             ", lmk4=" + fmt(lmk4.Shape) + " size=" + lmk4.Size +
                                             ", lmk5=" + fmt(lmk5.Shape) + " size=" + lmk5.Size);
                }
            }
            catch { }

            // flatten and concatenate along size axis to form unified tensors
            var clsConcat = ConcatFlatten(new Tensor[] { cls3, cls4, cls5 }, ctx);
            var regConcat = ConcatFlatten(new Tensor[] { reg3, reg4, reg5 }, ctx);
            var lmkConcat = ConcatFlatten(new Tensor[] { lmk3, lmk4, lmk5 }, ctx);
            int[][] shapes = new int[][] { cls3.Shape, cls4.Shape, cls5.Shape };
            return (clsConcat, regConcat, lmkConcat, shapes);
        }

        private Tensor DownsampleBy2(Tensor src, ComputacaoContexto ctx)
        {
            var s = src.Shape; int h = s[0], w = s[1], c = s[2];
            var fabrica = new Bionix.ML.nucleo.tensor.FabricaTensor(ctx);
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
            dst.GradFn = FabricaFuncoesRetropropagacao.CriarDownsample(ctx, src, dst);
            return dst;
        }

        private Tensor ConcatFlatten(Tensor[] parts, ComputacaoContexto ctx)
        {
            int total = 0; foreach (var p in parts) total += p.Size;
            var fabrica = new Bionix.ML.nucleo.tensor.FabricaTensor(ctx);
            var outT = fabrica.Criar(total);
            int idx = 0;
            foreach (var p in parts)
            {
                for (int i = 0; i < p.Size; i++) { outT[idx++] = p[i]; }
            }
            outT.RequiresGrad = true;
            outT.GradFn = FabricaFuncoesRetropropagacao.CriarConcat(ctx, parts, outT);
            return outT;
        }

        // Save/Load: enumerate known layer weights and save
        public void SaveWeights(string dir)
        {
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            // save all named parameters (weights + biases) and grads when present
            foreach (var kv in GetNamedParameters())
            {
                var name = kv.name;
                var t = kv.tensor;
                try
                {
                    if (t == null) continue;
                    var p = Path.Combine(dir, name + ".bin");
                    Bionix.ML.dados.serializacao.SerializadorTensor.SaveBinary(p, t);
                    if (t.Grad != null)
                    {
                        Bionix.ML.dados.serializacao.SerializadorTensor.SaveBinary(Path.Combine(dir, name + ".grad.bin"), t.Shape, t.Grad);
                    }
                }
                catch { }
            }
        }

        public void LoadWeights(string dir, bool loadGrad = false)
        {
            if (!Directory.Exists(dir)) return;
            foreach (var kv in GetNamedParameters())
            {
                var name = kv.name;
                var t = kv.tensor;
                try
                {
                    var p = Path.Combine(dir, name + ".bin");
                    if (!File.Exists(p) || t == null) continue;
                    var loaded = Bionix.ML.dados.serializacao.SerializadorTensor.LoadBinary(p);
                    if (loaded != null && t.Size == loaded.Size)
                    {
                        for (int i = 0; i < t.Size; i++) t[i] = loaded[i];
                    }
                    if (loadGrad)
                    {
                        var gpath = Path.Combine(dir, name + ".grad.bin");
                        if (File.Exists(gpath))
                        {
                            var g = Bionix.ML.dados.serializacao.SerializadorTensor.LoadBinary(gpath);
                            if (g != null && t.Grad != null && t.Grad.Length == g.Size)
                            {
                                var arr = g.ToArray();
                                for (int k = 0; k < t.Grad.Length; k++) t.Grad[k] = arr[k];
                            }
                        }
                    }
                }
                catch { }
            }
        }

        // Return all named parameters (base names used for saving files)
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
            // fpn
            yield return ("fpn_c5", TryGetNestedTensor(Pyramid, "LatC5", "Weight"));
            yield return ("fpn_c5.bias", TryGetNestedTensor(Pyramid, "LatC5", "Bias"));
            yield return ("fpn_c4", TryGetNestedTensor(Pyramid, "LatC4", "Weight"));
            yield return ("fpn_c4.bias", TryGetNestedTensor(Pyramid, "LatC4", "Bias"));
            yield return ("fpn_c3", TryGetNestedTensor(Pyramid, "LatC3", "Weight"));
            yield return ("fpn_c3.bias", TryGetNestedTensor(Pyramid, "LatC3", "Bias"));
            // heads p3/p4/p5 (include landmark convs and biases)
            yield return ("head_p3_conv", TryGetNestedTensor(HeadP3, "HeadConv", "Weight"));
            yield return ("head_p3_conv.bias", TryGetNestedTensor(HeadP3, "HeadConv", "Bias"));
            yield return ("head_p3_cls", TryGetNestedTensor(HeadP3, "ClsConv", "Weight"));
            yield return ("head_p3_cls.bias", TryGetNestedTensor(HeadP3, "ClsConv", "Bias"));
            yield return ("head_p3_reg", TryGetNestedTensor(HeadP3, "RegConv", "Weight"));
            yield return ("head_p3_reg.bias", TryGetNestedTensor(HeadP3, "RegConv", "Bias"));
            yield return ("head_p3_lmk", TryGetNestedTensor(HeadP3, "LandmarksConv", "Weight"));
            yield return ("head_p3_lmk.bias", TryGetNestedTensor(HeadP3, "LandmarksConv", "Bias"));

            yield return ("head_p4_conv", TryGetNestedTensor(HeadP4, "HeadConv", "Weight"));
            yield return ("head_p4_conv.bias", TryGetNestedTensor(HeadP4, "HeadConv", "Bias"));
            yield return ("head_p4_cls", TryGetNestedTensor(HeadP4, "ClsConv", "Weight"));
            yield return ("head_p4_cls.bias", TryGetNestedTensor(HeadP4, "ClsConv", "Bias"));
            yield return ("head_p4_reg", TryGetNestedTensor(HeadP4, "RegConv", "Weight"));
            yield return ("head_p4_reg.bias", TryGetNestedTensor(HeadP4, "RegConv", "Bias"));
            yield return ("head_p4_lmk", TryGetNestedTensor(HeadP4, "LandmarksConv", "Weight"));
            yield return ("head_p4_lmk.bias", TryGetNestedTensor(HeadP4, "LandmarksConv", "Bias"));

            yield return ("head_p5_conv", TryGetNestedTensor(HeadP5, "HeadConv", "Weight"));
            yield return ("head_p5_conv.bias", TryGetNestedTensor(HeadP5, "HeadConv", "Bias"));
            yield return ("head_p5_cls", TryGetNestedTensor(HeadP5, "ClsConv", "Weight"));
            yield return ("head_p5_cls.bias", TryGetNestedTensor(HeadP5, "ClsConv", "Bias"));
            yield return ("head_p5_reg", TryGetNestedTensor(HeadP5, "RegConv", "Weight"));
            yield return ("head_p5_reg.bias", TryGetNestedTensor(HeadP5, "RegConv", "Bias"));
            yield return ("head_p5_lmk", TryGetNestedTensor(HeadP5, "LandmarksConv", "Weight"));
            yield return ("head_p5_lmk.bias", TryGetNestedTensor(HeadP5, "LandmarksConv", "Bias"));
        }
    }
}
