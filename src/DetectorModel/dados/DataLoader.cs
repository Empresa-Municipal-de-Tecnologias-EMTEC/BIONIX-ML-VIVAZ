using System;
using System.Collections.Generic;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Bionix.ML.dados.imagem;
using Bionix.ML.dados.imagem.bmp;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.computacao;

namespace DetectorModel.dados
{
    public class Annotation
    {
        public string ImagePath { get; set; }
        public List<Box> Boxes { get; set; } = new List<Box>();
    }

    public struct Box
    {
        public int X { get; set; }
        public int Y { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public Box(int x, int y, int w, int h) { X = x; Y = y; Width = w; Height = h; }
    }

    public class DataLoader
    {
        private readonly string _annotationsFile;
        private readonly string _imagesRoot;

        public DataLoader(string annotationsFile, string imagesRoot)
        {
            _annotationsFile = annotationsFile ?? throw new ArgumentNullException(nameof(annotationsFile));
            _imagesRoot = imagesRoot ?? throw new ArgumentNullException(nameof(imagesRoot));
        }

        public class Sample
        {
            public Tensor Tensor { get; set; }
            public List<Box> Boxes { get; set; }
            public string ImagePath { get; set; }
        }

        // Convert an Annotation to a Sample by loading the image and transforming to Tensor
        public Sample AnnotationToSample(Annotation ann, ComputacaoContexto ctx)
        {
            if (ann == null) throw new ArgumentNullException(nameof(ann));
            if (ctx == null) throw new ArgumentNullException(nameof(ctx));

            if (!File.Exists(ann.ImagePath)) return null;

            // Use ManipuladorDeImagem to load into BMP and transform to Tensor
            BMP bmp;
            try
            {
                bmp = ManipuladorDeImagem.carregarBmpDeJPEG(ann.ImagePath);
            }
            catch
            {
                try { bmp = ManipuladorDeImagem.carregarBMPDePNG(ann.ImagePath); }
                catch { return null; }
            }

            var tensor = ManipuladorDeImagem.transformarEmTensor(bmp, ctx);
            return new Sample { Tensor = tensor, Boxes = ann.Boxes, ImagePath = ann.ImagePath };
        }

        // Parse WIDER FACE-style annotations (simple parser tolerant to variations)
        public IEnumerable<Annotation> ReadAnnotations()
        {
            if (!File.Exists(_annotationsFile)) yield break;

            var lines = File.ReadAllLines(_annotationsFile);
            int i = 0;
            while (i < lines.Length)
            {
                var line = lines[i].Trim();
                i++;
                if (string.IsNullOrWhiteSpace(line)) continue;

                string imageRel = line.Replace("/", Path.DirectorySeparatorChar.ToString());
                if (i >= lines.Length) break;
                if (!int.TryParse(lines[i].Trim(), out int faceCount)) break;
                i++;

                var ann = new Annotation();
                ann.ImagePath = Path.Combine(_imagesRoot, imageRel);

                for (int f = 0; f < faceCount && i < lines.Length; f++, i++)
                {
                    var parts = lines[i].Trim().Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length < 4) continue;
                    if (int.TryParse(parts[0], out int x) && int.TryParse(parts[1], out int y) &&
                        int.TryParse(parts[2], out int w) && int.TryParse(parts[3], out int h))
                    {
                        ann.Boxes.Add(new Box(x, y, w, h));
                    }
                }

                yield return ann;
            }
        }

        // Simple minibatch generator: returns lists of Annotation objects
        public IEnumerable<List<Annotation>> GetBatches(int batchSize)
        {
            var buffer = new List<Annotation>(batchSize);
            foreach (var ann in ReadAnnotations())
            {
                buffer.Add(ann);
                if (buffer.Count >= batchSize)
                {
                    yield return new List<Annotation>(buffer);
                    buffer.Clear();
                }
            }
            if (buffer.Count > 0) yield return buffer;
        }

        // Minibatch generator that returns tensors (uses provided computation context)
        public IEnumerable<List<Sample>> GetBatchesTensors(int batchSize, ComputacaoContexto ctx)
        {
            var buffer = new List<Sample>(batchSize);
            foreach (var ann in ReadAnnotations())
            {
                var sample = AnnotationToSample(ann, ctx);
                if (sample == null) continue;
                buffer.Add(sample);
                if (buffer.Count >= batchSize)
                {
                    yield return new List<Sample>(buffer);
                    buffer.Clear();
                }
            }
            if (buffer.Count > 0) yield return buffer;
        }
    }
}
