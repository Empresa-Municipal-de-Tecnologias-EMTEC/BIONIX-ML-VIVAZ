import model_detector as model_pkg
import bionix_ml.computacao.adaptadores.contexto as contexto_defs
import retina.retina_model as model_mod
import os

fn main() raises -> None:
    print("Iniciando treino Retina minimal...")
    var ctx = contexto_defs.criar_contexto_padrao("cpu")
    # create RetinaFace wrapper (internally builds the bloco)
    var params = model_mod.BlocoRetinaFaceParametros(320, 6, 3, 3, "cpu", 32, 16, 0.01, 0.5)
    var detector = model_mod.RetinaFace(params, os.path.join("MODELO", "retina_modelo"))
    var result = detector.treinar("DATASET", 320, 320, 32, 3, 0.0001, 2)
    print("Treinador finalizou:", result)
