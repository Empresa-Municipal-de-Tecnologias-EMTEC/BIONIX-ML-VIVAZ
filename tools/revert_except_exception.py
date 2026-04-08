#!/usr/bin/env python3
import os, re
root = os.path.join(os.path.dirname(__file__), '..', 'src', 'retina')
root = os.path.normpath(root)
changed = []
# simple replacement using read/write
for dirpath, dirs, files in os.walk(root):
    for fn in files:
        if not fn.endswith('.mojo'):
            continue
        path = os.path.join(dirpath, fn)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                s = f.read()
        except Exception as e:
            print('skip', path, e)
            continue
        new = s.replace('except Exception:', 'except _:')
        if new != s:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new)
            changed.append(path)
print('FILES_REVERTED:' + ','.join(changed))
