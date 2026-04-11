using System;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
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
        // Each landmarks entry is an array of 10 values: lx1,ly1,...,lx5,ly5
        public List<double[]> Landmarks { get; set; } = new List<double[]>();
        // source of the box: "landmarks" or "bbox"
        public string BoxSource { get; set; }
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
            public List<double[]> Landmarks { get; set; }
            public string BoxSource { get; set; }
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
            return new Sample { Tensor = tensor, Boxes = ann.Boxes, ImagePath = ann.ImagePath, Landmarks = ann.Landmarks, BoxSource = ann.BoxSource };
        }

        // Parse annotations. Supports WIDER-style txt (legacy) and CelebA CSV landmarks/bbox files.
        public IEnumerable<Annotation> ReadAnnotations()
        {
            if (!File.Exists(_annotationsFile)) yield break;

            // Detect CelebA CSV (landmarks) by header
            var firstLine = File.ReadLines(_annotationsFile).FirstOrDefault();
            if (firstLine != null && firstLine.Contains("lefteye_x"))
            {
                // Parse landmarks CSV: image_id,lefteye_x,lefteye_y,...,rightmouth_y
                var lmLines = File.ReadAllLines(_annotationsFile);
                var lmMap = new Dictionary<string, double[]>();
                for (int i = 1; i < lmLines.Length; i++)
                {
                    var parts = lmLines[i].Split(',');
                    if (parts.Length < 11) continue;
                    var img = parts[0].Trim();
                    var arr = new double[10];
                    for (int k = 0; k < 10; k++)
                    {
                        double v = 0.0; double.TryParse(parts[1 + k], NumberStyles.Any, CultureInfo.InvariantCulture, out v);
                        arr[k] = v;
                    }
                    lmMap[img] = arr;
                }

                // Try to load bbox CSV in same folder (list_bbox_celeba.csv)
                var annDir = Path.GetDirectoryName(_annotationsFile) ?? string.Empty;
                var bboxPath = Path.Combine(annDir, "list_bbox_celeba.csv");
                var bboxMap = new Dictionary<string, Box>();
                if (File.Exists(bboxPath))
                {
                    var bbLines = File.ReadAllLines(bboxPath);
                    for (int i = 1; i < bbLines.Length; i++)
                    {
                        var p = bbLines[i].Split(',');
                        if (p.Length < 5) continue;
                        var id = p[0].Trim();
                        if (int.TryParse(p[1], out int x) && int.TryParse(p[2], out int y) && int.TryParse(p[3], out int w) && int.TryParse(p[4], out int h))
                        {
                            bboxMap[id] = new Box(x, y, w, h);
                        }
                    }
                }

                foreach (var kv in lmMap)
                {
                    var ann = new Annotation();
                    ann.ImagePath = Path.Combine(_imagesRoot, kv.Key);
                    // If landmarks present, construct a GT bbox from landmarks (better fit for aligned images)
                    var lm = kv.Value;
                    if (lm != null && lm.Length >= 10)
                    {
                        double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
                        for (int k = 0; k < 5; k++)
                        {
                            double lx = lm[k * 2 + 0];
                            double ly = lm[k * 2 + 1];
                            if (lx < minX) minX = lx;
                            if (ly < minY) minY = ly;
                            if (lx > maxX) maxX = lx;
                            if (ly > maxY) maxY = ly;
                        }
                        // add small padding
                        int pad = 8;
                        int x = Math.Max(0, (int)Math.Floor(minX) - pad);
                        int y = Math.Max(0, (int)Math.Floor(minY) - pad);
                        int w = Math.Max(1, (int)Math.Ceiling(maxX) - x + pad);
                        int h = Math.Max(1, (int)Math.Ceiling(maxY) - y + pad);
                        ann.Boxes.Add(new Box(x, y, w, h));
                        ann.BoxSource = "landmarks";
                    }
                    else if (bboxMap.TryGetValue(kv.Key, out var bb))
                    {
                        ann.Boxes.Add(bb);
                        ann.BoxSource = "bbox";
                    }
                    ann.Landmarks.Add(kv.Value);
                    yield return ann;
                }

                yield break;
            }

            // Only CelebA CSV format is supported now. For other formats, no annotations are returned.
            yield break;
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
        // startSample: number of samples to skip (useful for resuming)
        public IEnumerable<List<Sample>> GetBatchesTensors(int batchSize, ComputacaoContexto ctx, int startSample = 0)
        {
            var buffer = new List<Sample>(batchSize);
            int globalIndex = 0;
            foreach (var ann in ReadAnnotations())
            {
                if (globalIndex++ < startSample) continue;
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
