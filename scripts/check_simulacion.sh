#!/bin/sh
# Chequeos previos de `make simular` y `make simulacion` (ver src/simulacion/README.md).
#
# Uso: scripts/check_simulacion.sh python|snapshot <dir>|latex
#   python    Python >= 3.12 con las dependencias de requirements-dev.txt
#   snapshot  existe el snapshot de datos <dir>
#   latex     pdflatex y los paquetes que usa reporte.tex (sale con 1 si falta algo)
# El Python se elige con PYTHON (default: python).

PYTHON=${PYTHON:-python}
errores=0

falla() {
    echo "ERROR: $1"
    shift
    for linea in "$@"; do echo "  $linea"; done
    errores=1
}

check_python() {
    if ! command -v "$PYTHON" >/dev/null 2>&1; then
        falla "No se encuentra '$PYTHON': instalar Python 3.12 (o pasar PYTHON=python3.12)." \
            "Ubuntu: sudo apt install python3.12 python3.12-venv" \
            "Fedora: sudo dnf install python3.12"
        return
    fi
    if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null; then
        falla "Hace falta Python 3.12 o mas nuevo ('$PYTHON' es $("$PYTHON" -V 2>&1))." \
            "Pasar otro con PYTHON=python3.12, o usar un venv / conda con 3.12."
        return
    fi
    # Importar los modulos de la simulacion delata cualquier dependencia faltante.
    faltante=$("$PYTHON" -c '
import importlib
try:
    for m in ("src.simulacion.un_modelo", "src.simulacion.simulador", "src.simulacion.reporte"):
        importlib.import_module(m)
except ModuleNotFoundError as exc:
    print(exc.name)
' 2>/dev/null)
    if [ -n "$faltante" ]; then
        falla "Falta instalar el paquete de Python '$faltante'." \
            "Instalar las dependencias con 'make install' (pip install -r requirements-dev.txt)."
    fi
}

check_snapshot() {
    if [ ! -f "$1/meta.json" ]; then
        falla "No existe el snapshot de datos $1." \
            "Bajarlo con 'make snapshot-collector' (con el collector local levantado en :3000)," \
            "o pasar otro con SNAPSHOT=<dir>."
    fi
}

# nombre:archivo que lo delata:paquete de Ubuntu (apt):paquete de Fedora (dnf)
PAQUETES_TEX="
inputenc:inputenc.sty:texlive-latex-base:texlive-latex
fontenc:fontenc.sty:texlive-latex-base:texlive-latex
babel:babel.sty:texlive-latex-base:texlive-babel
babel-spanish:spanish.ldf:texlive-lang-spanish:texlive-babel-spanish
geometry:geometry.sty:texlive-latex-recommended:texlive-geometry
booktabs:booktabs.sty:texlive-latex-recommended:texlive-booktabs
longtable:longtable.sty:texlive-latex-base:texlive-tools
graphicx:graphicx.sty:texlive-latex-base:texlive-graphics
float:float.sty:texlive-latex-recommended:texlive-float
ec:ecrm1000.tfm:texlive-latex-recommended:texlive-ec
"

check_latex() {
    if ! command -v pdflatex >/dev/null 2>&1; then
        falla "Falta instalar pdflatex (el reporte.tex se genera igual, pero no se compila el PDF)." \
            "Ubuntu: sudo apt install texlive-latex-base texlive-latex-recommended texlive-lang-spanish" \
            "Fedora: sudo dnf install texlive-latex texlive-babel texlive-babel-spanish texlive-geometry texlive-booktabs texlive-tools texlive-graphics texlive-float texlive-ec"
        return
    fi
    apt=""
    dnf=""
    for entrada in $PAQUETES_TEX; do
        nombre=${entrada%%:*}
        resto=${entrada#*:}
        archivo=${resto%%:*}
        resto=${resto#*:}
        if ! kpsewhich "$archivo" >/dev/null 2>&1; then
            echo "ERROR: Falta instalar $nombre (paquete de LaTeX que usa reporte.tex)."
            case " $apt " in *" ${resto%%:*} "*) ;; *) apt="$apt ${resto%%:*}" ;; esac
            case " $dnf " in *" ${resto#*:} "*) ;; *) dnf="$dnf ${resto#*:}" ;; esac
        fi
    done
    if [ -n "$apt" ]; then
        echo "  Ubuntu: sudo apt install$apt"
        echo "  Fedora: sudo dnf install$dnf"
        errores=1
    fi
}

case "$1" in
    python) check_python ;;
    snapshot) check_snapshot "$2" ;;
    latex) check_latex ;;
    *) echo "Uso: $0 python|snapshot <dir>|latex"; exit 2 ;;
esac
exit $errores
