using Bionix.ML.nucleo.tensor;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.computacao;
using Bionix.ML.dados;
using Bionix.ML.camadas;

namespace DetectorModel.modelo
{
    // Minimal RetinaFace skeleton to be expanded. Uses layers from DetectorModel.camada
    public class RetinaFaceModel
    {
        public ConvLayer Stem { get; }
        public BatchNormLayer StemBN { get; }
        public ActivationLayer StemAct { get; }
        public ConvLayer HeadCls { get; }
        public ConvLayer HeadReg { get; }
        // weight tensors (simple placeholders)
        public Tensor BackboneWeight { get; private set; }
        public Tensor HeadClsWeight { get; private set; }
        public Tensor HeadRegWeight { get; private set; }

        public RetinaFaceModel()
        {
            Stem = new ConvLayer(inChannels: 3, outChannels: 32, kernelSize: 3);
            StemBN = new BatchNormLayer(32);
            StemAct = new ActivationLayer(Bionix.ML.nucleo.funcoesAtivacao.ReLU.ReLU.Forward);
            HeadCls = new ConvLayer(inChannels: 32, outChannels: 1, kernelSize: 1);
            HeadReg = new ConvLayer(inChannels: 32, outChannels: 4, kernelSize: 1);
        }

        public void InitializeWeights(ComputacaoContexto ctx)
        {
            var fabrica = new FabricaTensor(ctx);
            // simple shapes: [out, in, k, k] or flattened
            BackboneWeight = fabrica.Criar(32, 3, 3, 3);
            HeadClsWeight = fabrica.Criar(1, 32);
            HeadRegWeight = fabrica.Criar(4, 32);
            // initialize conv layers and heads
            Stem.Initialize(ctx);
            StemBN.Initialize(ctx);
            HeadCls.Initialize(ctx);
            HeadReg.Initialize(ctx);
        }

        // Return tuple: (cls, reg)
        public (Tensor cls, Tensor reg) Forward(Tensor input, ComputacaoContexto ctx)
        {
            var x = Stem.Forward(input, ctx);
            x = StemBN.Forward(x, ctx);
            x = StemAct.Forward(x, ctx);
            var cls = HeadCls.Forward(x, ctx);
            var reg = HeadReg.Forward(x, ctx);
            return (cls, reg);
        }

        public void SaveWeights(string dir)
        {
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            if (BackboneWeight != null)
            {
                var p1 = Path.Combine(dir, "backbone.bin");
                Bionix.ML.dados.serializacao.SerializadorTensor.SaveBinary(p1, BackboneWeight);
                Bionix.ML.dados.serializacao.SerializadorTensor.SaveText(Path.ChangeExtension(p1, ".txt"), BackboneWeight);
            }
            if (HeadClsWeight != null)
            {
                var p2 = Path.Combine(dir, "head_cls.bin");
                Bionix.ML.dados.serializacao.SerializadorTensor.SaveBinary(p2, HeadClsWeight);
                Bionix.ML.dados.serializacao.SerializadorTensor.SaveText(Path.ChangeExtension(p2, ".txt"), HeadClsWeight);
            }
            if (HeadRegWeight != null)
            {
                var p3 = Path.Combine(dir, "head_reg.bin");
                Bionix.ML.dados.serializacao.SerializadorTensor.SaveBinary(p3, HeadRegWeight);
                Bionix.ML.dados.serializacao.SerializadorTensor.SaveText(Path.ChangeExtension(p3, ".txt"), HeadRegWeight);
            }
        }

        // Load weights from directory. If loadGrad is true and corresponding *.grad.bin exists, restore gradients.
        public void LoadWeights(string dir, bool loadGrad = false)
        {
            if (!Directory.Exists(dir)) return;
            var p1 = Path.Combine(dir, "backbone.bin");
            if (File.Exists(p1))
            {
                BackboneWeight = Bionix.ML.dados.serializacao.SerializadorTensor.LoadBinary(p1);
                if (loadGrad)
                {
                    var g1 = Path.Combine(dir, "backbone.grad.bin");
                    if (File.Exists(g1))
                    {
                        var gt = Bionix.ML.dados.serializacao.SerializadorTensor.LoadBinary(g1);
                        BackboneWeight.SetGrad(gt.ToArray());
                    }
                }
            }
            var p2 = Path.Combine(dir, "head_cls.bin");
            if (File.Exists(p2))
            {
                HeadClsWeight = Bionix.ML.dados.serializacao.SerializadorTensor.LoadBinary(p2);
                if (loadGrad)
                {
                    var g2 = Path.Combine(dir, "head_cls.grad.bin");
                    if (File.Exists(g2))
                    {
                        var gt = Bionix.ML.dados.serializacao.SerializadorTensor.LoadBinary(g2);
                        HeadClsWeight.SetGrad(gt.ToArray());
                    }
                }
            }
            var p3 = Path.Combine(dir, "head_reg.bin");
            if (File.Exists(p3))
            {
                HeadRegWeight = Bionix.ML.dados.serializacao.SerializadorTensor.LoadBinary(p3);
                if (loadGrad)
                {
                    var g3 = Path.Combine(dir, "head_reg.grad.bin");
                    if (File.Exists(g3))
                    {
                        var gt = Bionix.ML.dados.serializacao.SerializadorTensor.LoadBinary(g3);
                        HeadRegWeight.SetGrad(gt.ToArray());
                    }
                }
            }
        }
    }
}
