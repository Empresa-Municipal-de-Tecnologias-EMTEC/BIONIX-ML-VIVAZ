import retina.retina_model as model_mod
import bionix_ml.computacao.adaptadores.contexto as contexto_defs
import os

fn main() raises -> None:
    # Short tuning run: small epochs and low LR for quick feedback
    var epocas: Int = 5
    var early_stop: Bool = True
    print("Tuning run RetinaFace... epocas=", epocas, " early_stop=", early_stop)
    var params = model_mod.BlocoRetinaFaceParametros(320, 6, 3, 3, "cpu", 32, 16, 0.01, 0.5)
    var detector = model_mod.RetinaFace(params^, os.path.join("MODELO", "retina_modelo_tune"))
    # Reduced learning rate for stability
    var result = detector.treinar("DATASET", 320, 320, 32, epocas, 0.005, 8, 128, early_stop)
    print("Tuning run finalizado:", result)
