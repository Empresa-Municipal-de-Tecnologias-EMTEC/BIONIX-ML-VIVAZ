import bionix_ml.uteis.arquivo as arquivo_io

# Single movable instrumentation point. Call `debug_marker(desc, context)` to emit
# a single-line marker to /tmp/one_shot_debug.log. This file is intentionally
# lightweight and will not be invoked by default; we'll move calls to this
# function manually to focus debugging on one location at a time.

fn debug_marker(var desc: String, var context: String):
    try:
        var path = String("/tmp/one_shot_debug.log")
        var cur = arquivo_io.ler_texto_seguro(path)
        var out_line = String("")
        if len(cur) > 0:
            out_line = String(cur) + String("\n") + desc + String(" ") + context
        else:
            out_line = desc + String(" ") + context
        _ = arquivo_io.gravar_texto_seguro(path, out_line)
    except _:
        pass
