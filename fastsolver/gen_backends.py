"""Generate hyp_solver_cpu.py and hyp_solver_cuda.py from hyp_solver_template.py."""
import re, pathlib, hashlib
here = pathlib.Path(__file__).resolve().parent
src = (here / "hyp_solver_template.py").read_text()

def render(backend):
    text = src
    if backend == "cpu":
        text = text.replace("##IMPORTS##", "import numba\nfrom numba import njit, prange")
        text = text.replace("@DEVICE", '@njit(cache=True, error_model="numpy")')
        text = re.sub(r"##F64_2D\((\d+), (\d+)\)##", r"np.empty((\1, \2), dtype=np.float64)", text)
        text = re.sub(r"##F64_1D\((\d+)\)##", r"np.empty(\1, dtype=np.float64)", text)
        text = re.sub(r"##I64_1D\((\d+)\)##", r"np.empty(\1, dtype=np.int64)", text)
        text += '''

@njit(cache=True, parallel=True, error_model="numpy")
def solve_rows(thetas, grid, E, P, status, failidx, aset, nfev):
    n = thetas.shape[0]
    npts = grid.shape[0]
    for i in prange(n):
        st, j = solve_row(thetas[i], grid, npts, E[i], P[i], aset[i], nfev[i])
        status[i] = st
        failidx[i] = j


@njit(cache=True, error_model="numpy")
def solve_rows_serial(thetas, grid, E, P, status, failidx, aset, nfev):
    n = thetas.shape[0]
    npts = grid.shape[0]
    for i in range(n):
        st, j = solve_row(thetas[i], grid, npts, E[i], P[i], aset[i], nfev[i])
        status[i] = st
        failidx[i] = j
'''
    else:
        text = text.replace("##IMPORTS##", "import numba\nfrom numba import cuda")
        text = text.replace("@DEVICE", "@cuda.jit(device=True)")
        text = re.sub(r"##F64_2D\((\d+), (\d+)\)##", r"cuda.local.array((\1, \2), numba.float64)", text)
        text = re.sub(r"##F64_1D\((\d+)\)##", r"cuda.local.array(\1, numba.float64)", text)
        text = re.sub(r"##I64_1D\((\d+)\)##", r"cuda.local.array(\1, numba.int64)", text)
        text += '''

@cuda.jit
def solve_rows_kernel(thetas, grid, E, P, status, failidx, aset, nfev):
    i = cuda.grid(1)
    if i < thetas.shape[0]:
        npts = grid.shape[0]
        st, j = solve_row(thetas[i], grid, npts, E[i], P[i], aset[i], nfev[i])
        status[i] = st
        failidx[i] = j
'''
    text = text.replace("TEMPLATE (not importable)", f"GENERATED ({backend}) from hyp_solver_template.py sha256 {hashlib.sha256(src.encode()).hexdigest()[:16]}; do not edit")
    return text

for b in ("cpu", "cuda"):
    (here / f"hyp_solver_{b}.py").write_text(render(b))
    print("wrote", f"hyp_solver_{b}.py")
