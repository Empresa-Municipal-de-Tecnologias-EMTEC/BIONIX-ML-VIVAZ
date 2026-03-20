# BIONIX-ML-VIVAZ

<p align="center">
	<img src="ICONE.png" alt="Ícone do BIONIX-ML-VIVAZ" width="160">
</p>

Modelos e APIs para detecção e reconhecimento facial construídos sobre o framework **BIONIX-ML**.

Este repositório contém um modelo denominado *Vivaz* que expõe:

- uma API HTTP para inferência (endpoints para detecção e reconhecimento);
- pipes nomeados (named pipes) para integração local e passagem de imagens/atributos entre processos.

Importante: o repositório `BIONIX-ML` deve estar clonado como diretório irmão para que os imports `src.*` resolvam corretamente.

Repositório do framework principal: https://github.com/Empresa-Municipal-de-Tecnologias-EMTEC/BIONIX-ML

## Execução

1. Garanta o layout de diretórios ou, preferencialmente, declare a dependência via `pixi.toml` e importe `bionix_ml.*` nos seus módulos:

```
parent/
├─ BIONIX-ML/
└─ BIONIX-ML-VIVAZ/
```

Exemplo de `pixi.toml` do consumidor:

```toml
[dependencies]
bionix_ml = { path = "../BIONIX-ML" }
```

Exemplo de import (Mojo):

```mojo
import bionix_ml.computacao as computacao_pkg
```

Observação: apontar para `../BIONIX-ML/src` mantém compatibilidade com imports `src.*` legados, mas não é o fluxo recomendado — migre para `bionix_ml.*`.

2. Em WSL/Linux (recomendado) rode:

```bash
# compila comandos configurados no pixi
pixi run compilar

# executa o serviço e captura logs
pixi run executar > log.log 2>&1

# examinar logs
cat log.log
```

Observação: o projeto assume integração com módulos do `BIONIX-ML/src` — mantenha o repositório principal disponível no mesmo nível ou ajuste as dependências conforme descrito acima.

