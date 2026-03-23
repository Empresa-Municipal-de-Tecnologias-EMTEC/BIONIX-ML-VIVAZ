# Serviço API Mojo: compara duas imagens (registro vs câmera) e retorna match + score
import src.pre_processamento as preprocess_pkg
import src.extrair_embeddings as embeddings_pkg


def cosine_similarity(a, b):
    # naive cosine implementation for lists
    num = 0.0
    suma = 0.0
    sumb = 0.0
    for k in range(len(a)):
        var x = a[k]
        var y = b[k]
        num += x * y
        suma += x * x
        sumb += y * y
    if suma == 0 or sumb == 0:
        return 0.0
    return num / ((suma ** 0.5) * (sumb ** 0.5))


def comparar_imagens_bytes(db_bytes, cam_bytes):
    # db_bytes: image bytes loaded from DB (registered face)
    # cam_bytes: raw image bytes from camera
    face_db = preprocess_pkg.load_and_preprocess(db_bytes, 112)
    face_cam = preprocess_pkg.load_and_preprocess(cam_bytes, 112)
    emb_db = embeddings_pkg.extract_embedding(face_db)
    emb_cam = embeddings_pkg.extract_embedding(face_cam)
    score = cosine_similarity(emb_db, emb_cam)
    match = score >= 0.5
    return {"match": match, "score": float(score)}


def main():
    print("Serviço API (stub) pronto — use comparar_imagens_bytes(db, cam)")


# Entry point omitted for Mojo module usage; call `main()` explicitly when needed.
