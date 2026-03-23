# Helpers para salvar/ler metadados e pesos (txt + binário)
import os


def _ensure_parent_dir(path: String):
    try:
        var d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d)
    except Exception:
        pass


def save_metadata(path: String, meta: Any):
    _ensure_parent_dir(path)
    var f = open(path, 'w')
    for k in meta:
        try:
            var v = meta[k]
        except Exception:
            var v = ""
        f.write(String(k) + ": " + String(v) + "\n")
    f.close()


def save_weights(path: String, data: Any):
    _ensure_parent_dir(path)
    var f = open(path, 'wb')
    f.write(data)
    f.close()


def load_metadata(path: String):
    if not os.path.exists(path):
        return {}
    var meta = {}
    var f = open(path, 'r')
    for line in f:
        if ':' in line:
            var parts = line.split(':', 1)
            var k = parts[0].strip()
            var v = parts[1].strip()
            meta[k] = v
    f.close()
    return meta


def load_weights(path: String):
    if not os.path.exists(path):
        return None
    var f = open(path, 'rb')
    var data = f.read()
    f.close()
    return data
