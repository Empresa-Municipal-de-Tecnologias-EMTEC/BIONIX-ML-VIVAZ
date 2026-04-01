import bionix_ml.uteis.arquivo as arquivo_io
import os

# Simple centralized diagnostics logger. Globals are avoided to satisfy Mojo
# restrictions; logger is always enabled. If you need toggles, replace with
# build-time flags or function parameters.

fn _append_text(var path: String, var line: String) -> Bool:
    try:
        var cur = arquivo_io.ler_texto_seguro(path)
        var out = String("")
        if len(cur) > 0:
            out = String(cur) + String("\n") + line
        else:
            out = line
        _ = arquivo_io.gravar_texto_seguro(path, out)
        return True
    except _:
        return False

fn console_print(var *parts: String):
    try:
        var s = String("")
        for p in parts:
            if len(s) > 0:
                s = s + String(" ") + p
            else:
                s = p
        print(s)
    except _:
        pass

fn marker(var desc: String, var context: String):
    try:
        var path = String("/tmp/one_shot_debug.log")
        var line = desc + String(" ") + context
        _ = _append_text(path, line)
        try:
            console_print(String("[DBG-MARKER] ") + line)
        except _:
            pass
    except _:
        pass

fn marker_verbose(var desc: String, var context: String):
    try:
        var path = String("/tmp/one_shot_debug_verbose.log")
        var line = desc + String(" ") + context
        _ = _append_text(path, line)
        try:
            console_print(String("[DBG-MARKER-V] ") + line)
        except _:
            pass
    except _:
        pass

fn error_log(var desc: String, var context: String):
    try:
        var path = String("/tmp/one_shot_debug_error.log")
        var line = desc + String(" ") + context
        _ = _append_text(path, line)
        try:
            print("[ERR] " + line)
        except _:
            pass
    except _:
        pass
