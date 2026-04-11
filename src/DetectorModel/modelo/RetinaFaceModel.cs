using Bionix.ML.nucleo.tensor;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.computacao;
using Bionix.ML.grafo.CPU;
using Bionix.ML.dados;
using Bionix.ML.camadas;

namespace DetectorModel.modelo
{
    // RetinaFace-like model implemented with simple ResNet-style blocks, FPN and per-scale heads
    public class RetinaFaceModel
    {
        // backbone
        public ConvLayer Stem { get; }
        public ResidualBlock[] Stage2 { get; }
        public ResidualBlock[] Stage3 { get; }
        public ResidualBlock[] Stage4 { get; }

        // FPN
        public FPN Pyramid { get; }

        // per-scale heads
        public DetectionHead HeadP3 { get; }
        public DetectionHead HeadP4 { get; }
        public DetectionHead HeadP5 { get; }

        // Compatibility properties for existing training code
        public Tensor BackboneWeight => Stem?.Weight;
        public Tensor HeadClsWeight => HeadP3?.ClsConv?.Weight;
        public Tensor HeadRegWeight => HeadP3?.RegConv?.Weight;
        public Tensor HeadLmkWeight => HeadP3?.LandmarksConv?.Weight;
        public ConvLayer HeadCls => HeadP3?.ClsConv;
        public ConvLayer HeadReg => HeadP3?.RegConv;

        public RetinaFaceModel()
        {
            Stem = new ConvLayer(inChannels: 3, outChannels: 32, kernelSize: 3);
            // small residual stages
            Stage2 = new ResidualBlock[] { new ResidualBlock(32), new ResidualBlock(32) };
            Stage3 = new ResidualBlock[] { new ResidualBlock(32), new ResidualBlock(32) };
            Stage4 = new ResidualBlock[] { new ResidualBlock(32) };

            Pyramid = new FPN(inChannelsC5: 32, inChannelsC4: 32, inChannelsC3: 32, outChannels: 32);

            HeadP3 = new DetectionHead(inChannels: 32, interChannels: 32);
            HeadP4 = new DetectionHead(inChannels: 32, interChannels: 32);
            HeadP5 = new DetectionHead(inChannels: 32, interChannels: 32);
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

        // Create a simple concatenated output: flatten per-scale cls, reg and landmarks into single tensors
        public (Tensor cls, Tensor reg, Tensor lmk) Forward(Tensor input, ComputacaoContexto ctx)
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

            // flatten and concatenate along size axis to form unified tensors
            var clsConcat = ConcatFlatten(new Tensor[] { cls3, cls4, cls5 }, ctx);
            var regConcat = ConcatFlatten(new Tensor[] { reg3, reg4, reg5 }, ctx);
            var lmkConcat = ConcatFlatten(new Tensor[] { lmk3, lmk4, lmk5 }, ctx);
            return (clsConcat, regConcat, lmkConcat);
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
            dst.GradFn = new DownsampleFunction(src, dst);
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
            outT.GradFn = new ConcatFunction(parts, outT);
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
            yield return ("stem", Stem?.Weight);
            yield return ("stem.bias", Stem?.Bias);
            int i = 0;
            foreach (var b in Stage2)
            {
                yield return ($"stage2_{i}_conv1", b.Conv1?.Weight);
                yield return ($"stage2_{i}_conv1.bias", b.Conv1?.Bias);
                yield return ($"stage2_{i}_conv2", b.Conv2?.Weight);
                yield return ($"stage2_{i}_conv2.bias", b.Conv2?.Bias);
                i++;
            }
            i = 0;
            foreach (var b in Stage3)
            {
                yield return ($"stage3_{i}_conv1", b.Conv1?.Weight);
                yield return ($"stage3_{i}_conv1.bias", b.Conv1?.Bias);
                yield return ($"stage3_{i}_conv2", b.Conv2?.Weight);
                yield return ($"stage3_{i}_conv2.bias", b.Conv2?.Bias);
                i++;
            }
            i = 0;
            foreach (var b in Stage4)
            {
                yield return ($"stage4_{i}_conv1", b.Conv1?.Weight);
                yield return ($"stage4_{i}_conv1.bias", b.Conv1?.Bias);
                yield return ($"stage4_{i}_conv2", b.Conv2?.Weight);
                yield return ($"stage4_{i}_conv2.bias", b.Conv2?.Bias);
                i++;
            }
            // fpn
            yield return ("fpn_c5", Pyramid?.LatC5?.Weight);
            yield return ("fpn_c5.bias", Pyramid?.LatC5?.Bias);
            yield return ("fpn_c4", Pyramid?.LatC4?.Weight);
            yield return ("fpn_c4.bias", Pyramid?.LatC4?.Bias);
            yield return ("fpn_c3", Pyramid?.LatC3?.Weight);
            yield return ("fpn_c3.bias", Pyramid?.LatC3?.Bias);
            // heads p3/p4/p5 (include landmark convs and biases)
            yield return ("head_p3_conv", HeadP3?.HeadConv?.Weight);
            yield return ("head_p3_conv.bias", HeadP3?.HeadConv?.Bias);
            yield return ("head_p3_cls", HeadP3?.ClsConv?.Weight);
            yield return ("head_p3_cls.bias", HeadP3?.ClsConv?.Bias);
            yield return ("head_p3_reg", HeadP3?.RegConv?.Weight);
            yield return ("head_p3_reg.bias", HeadP3?.RegConv?.Bias);
            yield return ("head_p3_lmk", HeadP3?.LandmarksConv?.Weight);
            yield return ("head_p3_lmk.bias", HeadP3?.LandmarksConv?.Bias);

            yield return ("head_p4_conv", HeadP4?.HeadConv?.Weight);
            yield return ("head_p4_conv.bias", HeadP4?.HeadConv?.Bias);
            yield return ("head_p4_cls", HeadP4?.ClsConv?.Weight);
            yield return ("head_p4_cls.bias", HeadP4?.ClsConv?.Bias);
            yield return ("head_p4_reg", HeadP4?.RegConv?.Weight);
            yield return ("head_p4_reg.bias", HeadP4?.RegConv?.Bias);
            yield return ("head_p4_lmk", HeadP4?.LandmarksConv?.Weight);
            yield return ("head_p4_lmk.bias", HeadP4?.LandmarksConv?.Bias);

            yield return ("head_p5_conv", HeadP5?.HeadConv?.Weight);
            yield return ("head_p5_conv.bias", HeadP5?.HeadConv?.Bias);
            yield return ("head_p5_cls", HeadP5?.ClsConv?.Weight);
            yield return ("head_p5_cls.bias", HeadP5?.ClsConv?.Bias);
            yield return ("head_p5_reg", HeadP5?.RegConv?.Weight);
            yield return ("head_p5_reg.bias", HeadP5?.RegConv?.Bias);
            yield return ("head_p5_lmk", HeadP5?.LandmarksConv?.Weight);
            yield return ("head_p5_lmk.bias", HeadP5?.LandmarksConv?.Bias);
        }
    }
}
