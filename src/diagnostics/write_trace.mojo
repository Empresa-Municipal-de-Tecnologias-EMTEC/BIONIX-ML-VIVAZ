# Lightweight file-backed write trace for debugging OOB / memory-corruption symptoms
import bionix_ml.uteis.arquivo as arquivo_io

fn log_write(var desc: String):
    # Backwards-compatible simple write (keeps old behavior)
    try:
        log_write_ctx(desc, String(""), -1)
    except _:
        pass


fn log_write_ctx(var desc: String, var image_path: String, var a_idx: Int):
    try:
        var path = String("/tmp/bionix_write_trace.log")
        var cur = arquivo_io.ler_texto_seguro(path)
        var seq = 1
        if len(cur) > 0:
            var slices = cur.split("\n")
            seq = len(slices) + 1
        var prefix = String(seq) + String(" ")
        if len(image_path) > 0:
            prefix = prefix + String("img=") + image_path + String(" ")
        if a_idx >= 0:
            prefix = prefix + String("a_idx=") + String(a_idx) + String(" ")
        var out_line = prefix + desc
        var out_str = String("")
        if len(cur) > 0:
            out_str = String(cur) + String("\n") + out_line
        else:
            out_str = out_line
        _ = arquivo_io.gravar_texto_seguro(path, out_str)
    except _:
        pass

fn dump_recent(var n: Int) -> List[String]:
    var out = List[String]()
    try:
        var path = String("/tmp/bionix_write_trace.log")
        var cur = arquivo_io.ler_texto_seguro(path)
        if len(cur) == 0:
            return out^
        var slices = cur.split("\n")
        var start = len(slices) - n
        if start < 0:
            start = 0
        for i in range(start, len(slices)):
            out.append(String(slices[i]))
    except _:
        pass
    return out^
