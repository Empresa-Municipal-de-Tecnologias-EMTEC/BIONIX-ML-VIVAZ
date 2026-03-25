import os


fn _ensure_parent_dir(var path: String) -> None:
    try:
        var d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d)
    except _:
        pass


# Recebe metadados pré-formatados em texto e grava no arquivo.
fn save_metadata(var path: String, var meta_text: String) -> Bool:
    _ensure_parent_dir(path)
    try:
        var f = open(path, "w")
        f.write(meta_text)
        f.close()
        return True
    except _:
        return False


# Grava um blob binário (passar uma String/bytes ou List[Int] convertido para String)
fn save_weights(var path: String, var data: String) -> Bool:
    _ensure_parent_dir(path)
    try:
        var f = open(path, "wb")
        f.write(data)
        f.close()
        return True
    except _:
        return False


fn load_metadata(var path: String) -> String:
    try:
        if not os.path.exists(path):
            return ""
        var f = open(path, "r")
        var content = f.read()
        f.close()
        return content
    except _:
        return ""


fn load_weights(var path: String) -> String:
    try:
        if not os.path.exists(path):
            return ""
        var f = open(path, "rb")
        var data = f.read()
        f.close()
        return data
    except _:
        return ""
