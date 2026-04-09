import retina.retina_modelo as retina_modelo

# Thin wrapper module so callers importing `retina.retina_model`
# can access the implementation in `retina_modelo` without changing callers.
# Add simple passthroughs as needed by other scripts.

# Example passthroughs (add more if callers expect them):
fn carregar_modelo(path: String) -> None:
    try:
        retina_modelo.carregar_workspace(path)
    except _:
        pass

fn salvar_modelo(path: String) -> None:
    try:
        retina_modelo.salvar_workspace(path)
    except _:
        pass
