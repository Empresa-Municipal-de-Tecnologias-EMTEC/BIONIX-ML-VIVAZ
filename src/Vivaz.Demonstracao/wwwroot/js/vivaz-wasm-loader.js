// vivaz-wasm-loader.js (Legacy Proxy)
// Este arquivo agora apenas garante que a API window.vivazWasm esteja disponível,
// delegando a funcionalidade real para o vivaz.js unificado.
(function(){
  console.log("[vivaz-loader] Carregado. Delegando para vivaz.js...");
  // Se vivaz.js ainda não carregou, ele carregará em seguida.
  // Se já carregou, window.vivazWasm já existe.
})();
