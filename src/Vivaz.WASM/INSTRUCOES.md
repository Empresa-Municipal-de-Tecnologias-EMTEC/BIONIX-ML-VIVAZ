INSTRUÇÕES PARA PUBLICAR E USAR A VERSÃO WebAssembly (WASM)
===========================================================

Este documento agora descreve duas formas práticas de disponibilizar `Vivaz.WASM` para execução no navegador — a rota preferida (publicar um host/browser-wasm que produza `_framework`) e uma rota de contingência que usamos aqui quando o publish não gerou `_framework`: copiar os arquivos do runtime instalados pelo SDK (dotnet pack). Também contém recomendações operacionais (MIME, Git LFS).

Resumo das abordagens
---------------------
- Preferido: publicar um projeto host com `Microsoft.NET.Runtime.WebAssembly.Sdk` ou publicar `Vivaz.WASM` com `-r browser-wasm` de modo que o processo de `dotnet publish` gere o diretório `_framework` contendo `dotnet.js`, `dotnet.runtime.js`, `dotnet.native.wasm`, arquivos ICU e `blazor.boot.json`. Copie o conteúdo de `_framework` para `src/Vivaz.Demonstracao/wwwroot/vivaz-wasm`.
- Alternativa (contingência): quando o publish da biblioteca não gerar `_framework`, copie os arquivos runtime do SDK instalado (`C:\Program Files\dotnet\packs\Microsoft.NETCore.App.Runtime.Mono.browser-wasm\<vers>\runtimes\browser-wasm\native`) para `src/Vivaz.Demonstracao/wwwroot/vivaz-wasm` e crie um `blazor.boot.json` mínimo apontando para suas assemblies (modelo abaixo). Esta é a abordagem que foi usada aqui para tornar o demo funcional rapidamente.

Pré-requisitos
--------------
- SDK .NET 8 instalado e `dotnet` disponível no PATH.
- (recomendado) workload wasm-tools instalado:

```powershell
dotnet workload install wasm-tools
```

- Se o repositório usa Git LFS para grandes binários (por exemplo `Vivaz.WASM.dll`), execute antes:

```powershell
git lfs pull
```

Passo A — Publicação preferida (gera `_framework`)
-------------------------------------------------

1) Publicar para `browser-wasm` (exemplo):

```powershell
Push-Location BIONIX-ML-VIVAZ
dotnet publish src\Vivaz.WASM\Vivaz.WASM.csproj -c Release -r browser-wasm -o artifacts\vivaz_wasm_publish --no-self-contained
Pop-Location
```

2) Se o publish gerar um subdiretório `_framework`, copie seu conteúdo para a demo:

```powershell
robocopy "BIONIX-ML-VIVAZ\artifacts\vivaz_wasm_publish\_framework" "BIONIX-ML-VIVAZ\src\Vivaz.Demonstracao\wwwroot\vivaz-wasm" /MIR
```

3) Reinicie o servidor demo e faça um hard reload da página demo.

Copiar todo o conteúdo de `dist-wasm` para `vivaz-wasm`
----------------------------------------------------

Se você publicou para a pasta `dist-wasm` (ou usou `-o dist-wasm`), copie TODO o conteúdo dela para a pasta `vivaz-wasm` para que o demo use exatamente os artefatos gerados pelo publish. Exemplos:

PowerShell (Windows):

```powershell
Push-Location C:\PROJETOS\BIONIX\BIONIX-ML-VIVAZ
# remove qualquer conteúdo antigo
Remove-Item -Recurse -Force .\src\Vivaz.Demonstracao\wwwroot\vivaz-wasm\*

# copia tudo de dist-wasm para wwwroot/vivaz-wasm
robocopy .\dist-wasm .\src\Vivaz.Demonstracao\wwwroot\vivaz-wasm /MIR
Pop-Location
```

Linux/macOS (rsync):

```bash
rm -rf src/Vivaz.Demonstracao/wwwroot/vivaz-wasm/*
rsync -av --delete dist-wasm/ src/Vivaz.Demonstracao/wwwroot/vivaz-wasm/
```

Após copiar, confirme que `Vivaz.WASM.deps.json` (ou `blazor.boot.json`) e os artefatos runtime (`dotnet.js`, `dotnet.runtime.js`, `dotnet.native.wasm`, `dotnet.native.js`, `icudt.dat`, etc.) estão presentes em `src/Vivaz.Demonstracao/wwwroot/vivaz-wasm`.

Passo B — Contingência: copiar runtime do SDK (o que fizemos aqui)
----------------------------------------------------------------

Quando o publish da biblioteca não produzir `_framework`, os arquivos runtime necessários podem ser copiados diretamente a partir do pack do SDK instalado. Exemplo (o caminho pode variar conforme a versão):

```powershell
robocopy "C:\Program Files\dotnet\packs\Microsoft.NETCore.App.Runtime.Mono.browser-wasm\8.0.26\runtimes\browser-wasm\native" "src\Vivaz.Demonstracao\wwwroot\vivaz-wasm" dotnet.js dotnet.runtime.js dotnet.native.wasm icudt.dat icudt_EFIGS.dat icudt_CJK.dat icudt_no_CJK.dat /NFL /NDL /NJH /NJS /NC /NS
```

Após copiar os arquivos do runtime, crie um `blazor.boot.json` mínimo em `src\Vivaz.Demonstracao\wwwroot\vivaz-wasm` listando as assemblies e os recursos de runtime (um exemplo foi gerado automaticamente nesta máquina e está presente no repositório). O `blazor.boot.json` produzido manualmente deve listar pelo menos as assemblies usadas pelo `entryAssembly`.

Observação de segurança/consistência
----------------------------------
- Preferimos usar o `dotnet publish` para produzir o `blazor.boot.json` (contém os hashes e o layout corretos). O `blazor.boot.json` manual é um fallback rápido; para produção gere o manifest com `dotnet publish` sempre que possível.
- Se os DLLs no servidor vierem como ponteiros Git LFS (arquivos pequenos contendo metadados), o navegador receberá um arquivo inválido e o runtime falhará com erros/asserções. Execute `git lfs pull` para garantir os binários reais estejam presentes antes de copiar/servir.

MIME / servidor
----------------
Adicione mapeamentos de tipo de conteúdo no `Program.cs` do demo para garantir que `.dll` e `.wasm` sejam servidos corretamente:

```csharp
var contentTypeProvider = new FileExtensionContentTypeProvider();
contentTypeProvider.Mappings[".dll"] = "application/octet-stream";
contentTypeProvider.Mappings[".pdb"] = "application/octet-stream";
contentTypeProvider.Mappings[".wasm"] = "application/wasm";
app.UseStaticFiles(new StaticFileOptions { FileProvider = provider, RequestPath = "", ContentTypeProvider = contentTypeProvider });
```

Teste rápido
------------
Verifique que o servidor está servindo os arquivos do runtime e as assemblies:

```powershell
Invoke-WebRequest -Uri http://localhost:5000/vivaz-wasm/dotnet.js -Method Head | Format-List
Invoke-WebRequest -Uri http://localhost:5000/vivaz-wasm/Vivaz.WASM.dll -Method Head | Format-List
```

Procure `StatusCode: 200` e `Content-Type` apropriado.

Nota sobre `blazor.boot.json` manual
------------------------------------
Um `blazor.boot.json` manual (como o que geramos para teste) funciona como um curto-circuito para o loader do runtime, mas não contém os mesmos hashes/associações que o `dotnet publish` produz. Use-o apenas para desenvolvimento local rápido e prefira o manifest gerado por `dotnet publish` para entregas reais.

Detalhe importante: `dotnet.native.js` (JS module native)
-------------------------------------------------------
Em alguns ambientes o publish gera `dotnet.native.js` (um wrapper JS que acompanha o .wasm). Se esse arquivo não estiver presente no diretório `wwwroot/vivaz-wasm` o runtime reclamará com mensagens do tipo "Expect to have one dotnetwasm asset in resources" ou "Expect to have one js-module-native".

Como recuperar e instalar `dotnet.native.js` (passos):

1. Localize o pack do runtime no SDK (exemplo de caminho no Windows):

```
C:\Program Files\dotnet\packs\Microsoft.NETCore.App.Runtime.Mono.browser-wasm\<vers>\runtimes\browser-wasm\native
```

2. Copie `dotnet.native.js` para a pasta da demo `wwwroot/vivaz-wasm`. Exemplo (PowerShell / robocopy):

```powershell
robocopy "C:\Program Files\dotnet\packs\Microsoft.NETCore.App.Runtime.Mono.browser-wasm\8.0.26\runtimes\browser-wasm\native" "src\Vivaz.Demonstracao\wwwroot\vivaz-wasm" dotnet.native.js /NFL /NDL /NJH /NJS /NC /NS
```

3. Atualize `blazor.boot.json` para expor corretamente o asset `.wasm` e o módulo JS nativo. A estrutura compatível com .NET 8 que funciona é similar a:

```json
{
	"mainAssemblyName": "Vivaz.WASM.dll",
	"resources": {
		"assembly": {
			"Vivaz.WASM.dll": "",
			"Bionix.ML.dll": "",
			"DetectorLeveBModel.dll": "",
			"DetectorModel.dll": "",
			"IdentificadorLeveModel.dll": "",
			"ILGPU.dll": "",
			"SixLabors.ImageSharp.dll": ""
		},
		"runtime": {
			"icudt.dat": ""
		},
		"wasmNative": {
			"dotnet.native.wasm": ""
		},
		"jsModuleNative": {
			"dotnet.native.js": ""
		},
		"jsModuleRuntime": {
			"dotnet.runtime.js": ""
		}
	}
}
```

Observações:
- O nome exato das chaves importa: `wasmNative` e `js-module-native` são as chaves que o runtime .NET 8 procura.
- Os valores podem ser strings vazias durante desenvolvimento; para produção prefira o `blazor.boot.json` gerado por `dotnet publish` que contém hashes.
- Depois de copiar `dotnet.native.js` e atualizar o manifest, reinicie o servidor e faça um hard reload do navegador (Ctrl+F5).

Se quiser, eu executo o `robocopy` e atualizo o `blazor.boot.json` para você (já fiz aqui como parte das correções). Marquei essa etapa no checklist.

Checklist atualizado
--------------------
- [x] Remover arquivos manuais de `wwwroot/vivaz-wasm` (se necessário)
- [x] `dotnet workload install wasm-tools` (se necessário)
- [x] `dotnet publish -r browser-wasm` ou publicar host WebAssembly
- [x] Se `_framework` não existir, copiar runtime do SDK pack (fallback)
- [x] Copiar publish/_framework ou runtime SDK -> `src/Vivaz.Demonstracao/wwwroot/vivaz-wasm`
- [x] Verificar Git LFS (`git lfs pull`) antes de servir grandes DLLs
- [x] Verificar mapeamentos MIME em `Program.cs`
- [ ] Reiniciar o servidor demo e testar no navegador (Ctrl+F5)

Se quiser, eu posso reiniciar o servidor demo aqui e validar o boot no console do navegador — diga "reinicie e teste" e eu faço isso e coleto os logs.
