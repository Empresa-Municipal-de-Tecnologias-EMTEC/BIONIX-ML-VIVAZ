# Configurações centrais do POC
alias MODEL_DIR = "MODELO"
alias DATASET_ROOT = "DATASET"

# Como as imagens podem vir com proporções diferentes, usamos como referência
# o tamanho do lado menor (short side). Exemplos: 112, 160.
alias INPUT_SHORT_SIDE = 112

# Resolução máxima de entrada (maior dimensão permitida) para proteger memória
alias MAX_INPUT_RESOLUTION = 1024

# Número de identidades/amostras por época (ajustável)
alias IDENTITIES_PER_EPOCH = 5
alias IMAGES_PER_IDENTITY = 2

# Treino
alias EPOCHS = 30
alias MATCH_THRESHOLD = 0.5

# Se True, gera arquivos de retângulos de face (um arquivo por imagem,
# mesmo nome da imagem, extensão .box) antes do treino. Pode ser útil
# para criar pseudo-labels ou preparar anotações iniciais.
alias GERAR_RETANGULOS_FACE = False
