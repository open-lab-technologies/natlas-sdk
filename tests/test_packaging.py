import subprocess
import sys

HEAVY = ["fastapi", "uvicorn", "llama_cpp", "torch", "transformers", "numpy", "soundfile",
         "sacrebleu", "jiwer", "streamlit"]


def test_import_natlas_loads_no_optional_dependencies():
    code = (
        "import sys, natlas, json; "
        f"print(json.dumps([m for m in {HEAVY!r} if m in sys.modules]))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_public_api_is_exported():
    import natlas

    for name in ("NAtlas", "AsyncNAtlas", "NAtlasError", "RateLimitError", "get_language"):
        assert name in natlas.__all__
    assert natlas.__version__
