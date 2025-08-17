import os
import ssgetpy

def top_matches_by_rows(N: int, square: bool = True, prefer_ge: bool = True, top_k: int = 10):
    """
    Busca matrices por tamaño de filas N.
    - square=True  -> solo matrices cuadradas (nrows == ncols)
    - square=False -> matrices rectangulares o cuadradas (solo se filtra por nrows)
    - prefer_ge=True -> prioriza nrows >= N (la más chica por encima de N)
      prefer_ge=False -> prioriza la más cercana a N (arriba o abajo)
    Devuelve una lista con las mejores top_k coincidencias.
    """
    mats = ssgetpy.search()  # índice completo (rápido: ~3k matrices)
    # Filtro por cuadrada si aplica
    if square:
        mats = [m for m in mats if m.rows == m.cols]
    # Orden de preferencia
    if prefer_ge:
        # primero las que tienen nrows >= N, y dentro de esas la más chica;
        # si no hay >= N, caerá a las < N
        mats = sorted(mats, key=lambda m: (m.rows < N, abs(m.rows - N), m.cols))
    else:
        # la más cercana a N (arriba o abajo)
        mats = sorted(mats, key=lambda m: (abs(m.rows - N), abs(m.cols - (m.rows if square else m.cols))))
    return mats[:top_k]

def download_matrix_by_choice(N: int,
                              square: bool = True,
                              prefer_ge: bool = True,
                              choice_index: int = 0,
                              destpath: str = ".",
                              fmt: str = "MM"):
    """
    Encuentra candidatos por N, muestra un top y descarga el elegido (choice_index).
    """
    candidates = top_matches_by_rows(N, square=square, prefer_ge=prefer_ge, top_k=10)
    if not candidates:
        raise RuntimeError("No se encontraron matrices que cumplan el criterio.")

    print(f"Top candidatos para N={N} (square={square}, prefer_ge={prefer_ge}):")
    for i, m in enumerate(candidates):
        shape = f"{m.rows}x{m.cols}"
        print(f"[{i}] {m.group}/{m.name:20s}  shape={shape:>12}  nnz={m.nnz}")

    idx = max(0, min(choice_index, len(candidates)-1))
    sel = candidates[idx]
    print(f"\nDescargando: {sel.group}/{sel.name}  ({m.rows}x{m.cols}, nnz={sel.nnz})")

    res = ssgetpy.fetch(group=sel.group, name=sel.name, format=fmt)
    out = res.download(format=fmt, extract=True, destpath=destpath)
    print("Guardado en:", out)
    return sel, out

# -----------------------
# USO:
# 1) MATRIZ CUADRADA cercana a N (prioriza nrows >= N)

"""sel, path = download_matrix_by_choice(
    N=5000,
    square=True,
    prefer_ge=True,     # primero >= N
    choice_index=0,     # elegís la #0 del top (podés cambiar)
    destpath=os.getcwd(),
    fmt="MM"
)
"""
# 2) MATRIZ NO CUADRADA (NxK): con elegir N me alcanza → square=False
sel2, path2 = download_matrix_by_choice(
    N=3000,
    square=False,
    prefer_ge=False,   # la más cercana (arriba o abajo)
    choice_index=0,
    destpath=os.getcwd(),
)
