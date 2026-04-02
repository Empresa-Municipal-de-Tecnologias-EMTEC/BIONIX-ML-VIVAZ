import struct, os

path = 'MODELO/retina_modelo/cnn_bias_saida.tensor.txt'
if os.path.exists(path):
    txt = open(path).read()
    print('cnn_bias_saida.tensor.txt (first 300 chars):', repr(txt[:300]))

path2 = 'MODELO/retina_modelo/bias_reg.bin'
if os.path.exists(path2):
    d = open(path2,'rb').read()
    n = len(d)//4
    vals = struct.unpack_from('<'+'f'*min(n,8), d)
    print('bias_reg.bin vals:', [f'{v:.4f}' for v in vals])

path3 = 'MODELO/retina_modelo/retina_state.txt'
if os.path.exists(path3):
    print('retina_state.txt:', open(path3).read())
