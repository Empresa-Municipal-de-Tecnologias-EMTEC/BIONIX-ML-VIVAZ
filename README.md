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

1. Garanta o layout de diretórios:

```
parent/
├─ BIONIX-ML/
└─ BIONIX-ML-VIVAZ/
```

2. Na sessão Linux/WSL, instale os pré-requisitos do `pixi`/`mojo` e execute o serviço de inferência conforme instruções em `src/`.

Observação: o projeto assume integração com módulos do `BIONIX-ML/src` — mantenha o repositório principal disponível no mesmo nível.

