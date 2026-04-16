Guia rápido para adicionar arquivos grandes ao Git usando Git LFS

1) Instalar Git LFS (se ainda não instalado):
   - Windows: baixe e instale de https://git-lfs.github.com/ ou use `choco install git-lfs`.

2) Inicializar Git LFS no repositório (execute na raiz do repositório):
   ```powershell
   git lfs install
   ```

3) Adicionar padrões de arquivos grandes ao LFS (exemplos):
   ```powershell
   git lfs track "**/vivaz-wasm/*.wasm"
   git lfs track "**/vivaz-wasm/*.data"
   git lfs track "**/vivaz-wasm/*.js"
   git lfs track "PESOS/**"
   git lfs track "**/*.bin"
   ```

   Isso cria (ou atualiza) o arquivo `.gitattributes` com as regras.

4) Commit dos arquivos .gitattributes e dos artefatos LFS
   ```powershell
   git add .gitattributes
   git commit -m "Track wasm/runtime/weights with Git LFS"
   git add <arquivos grandes ou pasta release>
   git commit -m "Add release wasm artifacts (via LFS)"
   git push
   ```

Observações:
- Tenha certeza de que seu servidor remoto (GitHub, GitLab, Bitbucket) aceita Git LFS e você tem quota disponível.
- Se preferir não commitar `wwwroot` na branch principal, considere publicar os artefatos em uma release/distribution storage (S3, Azure Blob) e manter apenas o código fonte no repositório.
