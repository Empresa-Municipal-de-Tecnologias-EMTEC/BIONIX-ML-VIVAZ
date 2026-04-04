import retina.retina_model as model_mod
import bionix_ml.computacao.adaptadores.contexto as contexto_defs
import os

fn main() raises -> None:
    # ── Configuração do treino ──────────────────────────────────────────────
    # For quick debug/tuning we use a small number of epochs and early stop
    var epocas: Int = 1000         # número máximo de épocas (debug)
    var early_stop: Bool = False    # para automaticamente ao convergir
    # ───────────────────────────────────────────────────────────────────────
    print("Iniciando treino RetinaFace... epocas=", epocas, " early_stop=", early_stop)
    var params = model_mod.BlocoRetinaFaceParametros(320, 6, 3, 3, "cpu", 32, 16, 0.01, 0.5)
    var detector = model_mod.RetinaFace(params^, os.path.join("MODELO", "retina_modelo"))
    # Reduced LR for stability during tuning
    var result = detector.treinar("DATASET", 320, 320, 32, epocas, 0.005, 8, 128, early_stop)
    print("Treino finalizado:", result)
