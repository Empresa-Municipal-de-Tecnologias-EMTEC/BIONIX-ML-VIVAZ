using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Text.Json;
using Bionix.ML.computacao;
using Bionix.ML.nucleo.tensor;
using Bionix.ML.dados.imagem;
using IdentificadorModel.modelo;

namespace IdentificadorModel
{
    public class ExecutarIdentificador
    {
        public class EmbeddingRecord
        {
            public string Label { get; set; }
            public string ImagePath { get; set; }
            public double[] Embedding { get; set; }
        }

        public static void Main(string[] args)
        {
            if (args == null || args.Length == 0)
            {
                Console.WriteLine("Usage: enroll <label> <image> | enroll-folder <folder> | identify <image> [--top N] [--threshold T]");
                return;
            }

            var computeEnv = Environment.GetEnvironmentVariable("COMPUTE") ?? "SIMD";
            ComputacaoContexto ctx = computeEnv.Equals("CPU", StringComparison.OrdinalIgnoreCase) ? (ComputacaoContexto)new ComputacaoCPUContexto() : new ComputacaoCPUSIMDContexto();

            try
            {
                Run(args, ctx);
            }
            finally
            {
                if (ctx is IDisposable d) d.Dispose();
            }
        }

        private static void Run(string[] args, ComputacaoContexto ctx)
        {
            var pesosDir = Path.Combine(Directory.GetCurrentDirectory(), "PESOS", "IDENTIFICADOR");
            Directory.CreateDirectory(pesosDir);
            var dbPath = Path.Combine(pesosDir, "embeddings.json");

            var model = new ArcFaceModel(embeddingSize: 128, ctx: ctx);
            model.InitializeWeights(ctx);
            // attempt to load fc weights
            model.LoadWeights(pesosDir);

            var cmd = args[0].ToLowerInvariant();
            if (cmd == "enroll" && args.Length >= 3)
            {
                var label = args[1];
                var img = args[2];
                EnrollSingle(model, ctx, dbPath, label, img);
                model.SaveWeights(pesosDir);
                return;
            }
            else if (cmd == "enroll-folder" && args.Length >= 2)
            {
                var folder = args[1];
                EnrollFolder(model, ctx, dbPath, folder);
                model.SaveWeights(pesosDir);
                return;
            }
            else if (cmd == "identify" && args.Length >= 2)
            {
                var img = args[1];
                int top = 3;
                double threshold = 0.5;
                for (int i = 2; i < args.Length; i++) { if (args[i] == "--top" && i + 1 < args.Length) { int.TryParse(args[i + 1], out top); i++; } if (args[i] == "--threshold" && i + 1 < args.Length) { double.TryParse(args[i + 1], out threshold); i++; } }
                Identify(model, ctx, dbPath, img, top, threshold);
                return;
            }

            Console.WriteLine("Unknown command or missing arguments.");
        }

        private static List<EmbeddingRecord> LoadDb(string dbPath)
        {
            if (!File.Exists(dbPath)) return new List<EmbeddingRecord>();
            try { var json = File.ReadAllText(dbPath); return JsonSerializer.Deserialize<List<EmbeddingRecord>>(json) ?? new List<EmbeddingRecord>(); } catch { return new List<EmbeddingRecord>(); }
        }

        private static void SaveDb(string dbPath, List<EmbeddingRecord> db)
        {
            try { File.WriteAllText(dbPath, JsonSerializer.Serialize(db)); } catch { }
        }

        private static void EnrollSingle(ArcFaceModel model, ComputacaoContexto ctx, string dbPath, string label, string imagePath)
        {
            if (!File.Exists(imagePath)) { Console.WriteLine($"Image not found: {imagePath}"); return; }
            try
            {
                var bmp = ManipuladorDeImagem.carregarBmpDeJPEG(imagePath);
                var resized = ManipuladorDeImagem.redimensionar(bmp, 112, 112);
                var tensor = ManipuladorDeImagem.transformarEmTensor(resized, ctx);
                var embTensor = model.Forward(tensor, ctx);
                var emb = embTensor.ToArray();
                var db = LoadDb(dbPath);
                db.Add(new EmbeddingRecord { Label = label, ImagePath = imagePath, Embedding = emb });
                SaveDb(dbPath, db);
                Console.WriteLine($"Enrolled {imagePath} as '{label}' (db size={db.Count})");
            }
            catch (Exception ex) { Console.WriteLine($"Enroll failed: {ex.Message}"); }
        }

        private static void EnrollFolder(ArcFaceModel model, ComputacaoContexto ctx, string dbPath, string folder)
        {
            if (!Directory.Exists(folder)) { Console.WriteLine($"Folder not found: {folder}"); return; }
            var subdirs = Directory.GetDirectories(folder);
            int count = 0;
            foreach (var sd in subdirs)
            {
                var label = Path.GetFileName(sd);
                var images = Directory.GetFiles(sd).Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".png", StringComparison.OrdinalIgnoreCase)).ToArray();
                foreach (var im in images)
                {
                    EnrollSingle(model, ctx, dbPath, label, im);
                    count++;
                }
            }
            Console.WriteLine($"Enroll-folder processed {count} images.");
        }

        private static void Identify(ArcFaceModel model, ComputacaoContexto ctx, string dbPath, string imagePath, int top = 3, double threshold = 0.5)
        {
            if (!File.Exists(imagePath)) { Console.WriteLine($"Image not found: {imagePath}"); return; }
            var db = LoadDb(dbPath);
            if (db.Count == 0) { Console.WriteLine("Empty embedding DB. Enroll samples first."); return; }
            try
            {
                var bmp = ManipuladorDeImagem.carregarBmpDeJPEG(imagePath);
                var resized = ManipuladorDeImagem.redimensionar(bmp, 112, 112);
                var tensor = ManipuladorDeImagem.transformarEmTensor(resized, ctx);
                var embTensor = model.Forward(tensor, ctx);
                var emb = embTensor.ToArray();

                // assume stored embeddings are normalized; compute dot product (cosine)
                var scores = new List<(double score, EmbeddingRecord rec)>();
                foreach (var r in db)
                {
                    if (r.Embedding == null || r.Embedding.Length != emb.Length) continue;
                    double dot = 0.0; for (int i = 0; i < emb.Length; i++) dot += emb[i] * r.Embedding[i];
                    scores.Add((dot, r));
                }
                var best = scores.OrderByDescending(s => s.score).Take(top).ToArray();
                Console.WriteLine($"Top {best.Length} matches for {imagePath}:");
                foreach (var b in best) Console.WriteLine($" Label={b.rec.Label} Image={b.rec.ImagePath} Score={b.score:F4}");
                if (best.Length > 0 && best[0].score >= threshold) Console.WriteLine($"IDENTIFIED: {best[0].rec.Label} (score={best[0].score:F4})");
                else Console.WriteLine("No match above threshold.");
            }
            catch (Exception ex) { Console.WriteLine($"Identify failed: {ex.Message}"); }
        }
    }
}
