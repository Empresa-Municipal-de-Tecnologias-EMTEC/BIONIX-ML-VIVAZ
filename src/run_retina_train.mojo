import retina.model_detector as model_pkg
import bionix_ml.computacao.adaptadores.contexto as contexto_defs
import retina.retina_model as model_mod
import os

fn main() raises -> None:
    print("Iniciando treino Retina minimal...")
    # debug: quick marker and PID if available
    # PID not portable in this environment; print a simple marker
    print("[DEBUG] run_retina_train: start marker")
    var ctx = contexto_defs.criar_contexto_padrao("cpu")
    # create RetinaFace wrapper (internally builds the bloco)
    var params = model_mod.BlocoRetinaFaceParametros(320, 6, 3, 3, "cpu", 32, 16, 0.01, 0.5)
    print("[DEBUG] run_retina_train: antes de criar RetinaFace")
    var detector = model_mod.RetinaFace(params^, os.path.join("MODELO", "retina_modelo"))
    print("[DEBUG] run_retina_train: RetinaFace criado")
    print("[DEBUG] run_retina_train: chamando detector.treinar()")
    var result = detector.treinar("DATASET", 320, 320, 32, 3, 0.0001, 2)
    print("[DEBUG] run_retina_train: detector.treinar() retornou")
    print("Treinador finalizou:", result)
