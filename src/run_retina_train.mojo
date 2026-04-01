import retina.retina_model as model_mod
import bionix_ml.computacao.adaptadores.contexto as contexto_defs
import os

fn main() raises -> None:
    print("Iniciando treino RetinaFace...")
    var params = model_mod.BlocoRetinaFaceParametros(320, 6, 3, 3, "cpu", 32, 16, 0.01, 0.5)
    var detector = model_mod.RetinaFace(params^, os.path.join("MODELO", "retina_modelo"))
    var result = detector.treinar("DATASET", 320, 320, 32, 3, 0.0001, 4)
    print("Treino finalizado:", result)
