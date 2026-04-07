import retina.retina_modelo as model_mod
import bionix_ml.computacao.adaptadores.contexto as contexto_defs
import os

fn main() raises -> None:
    # ── Configuração do treino ──────────────────────────────────────────────
    # For quick debug/tuning we use a small number of epochs and early stop
    var epocas: Int = 1         # número máximo de épocas (debug) - smoke test
    var early_stop: Bool = False    # para automaticamente ao convergir
    # ───────────────────────────────────────────────────────────────────────
    print("Iniciando treino RetinaFace... epocas=", epocas, " early_stop=", early_stop)
    var params = model_mod.BlocoRetinaFaceParametros(320, 6, 3, 3, "cpu", 32, 16, 0.01, 0.5)
    var detector = model_mod.RetinaFace(params^, os.path.join("MODELO", "retina_modelo"))
    # Configure conv-FPN pipeline (placeholder scaffolding)
    try:
        _ = detector.configurar_conv_fpn("mobilenet_v2")
    except _:
        pass
    # Reduced LR for stability during tuning
    # train only on two annotated identities per user request
    var allowed = List[String]()
    allowed.append("n000002"); allowed.append("n000003")
    var result = detector.treinar("DATASET", 320, 320, 32, epocas, 0.05, 8, 128, early_stop, allowed)
    print("Treino finalizado:", result)
