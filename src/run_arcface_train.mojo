import reconhecedor.arcface_model as arc_model
import os

fn main() raises -> None:
    print("Iniciando treino ArcFace...")
    var params = arc_model.ArcFaceParametros(64, 8, 3, 3, 128, 0, "cpu")
    var modelo = arc_model.ArcFace(params^, os.path.join("MODELO", "arcface_modelo"))
    var result = modelo.treinar("DATASET", 1, 0.05, 8, 128)
    print("Treino finalizado:", result)
