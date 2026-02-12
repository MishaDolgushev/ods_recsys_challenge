import os
import tempfile
from pathlib import Path

def rectools_setup(
    tmp_dir: str = r"C:\rectools_tmp",
    patch_named_tempfile: bool = True,
    add_safe_globals: bool = True,
) -> None:
    """
    Fixes rectools (SASRec) save/load on Windows:
    - avoids NamedTemporaryFile lock issue
    - sets temp dir to writable location
    - allowlists numpy globals for torch.load(weights_only=True) behavior
    """
    tmp_path = Path(tmp_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)

    os.environ["TEMP"] = str(tmp_path)
    os.environ["TMP"] = str(tmp_path)
    os.environ["TMPDIR"] = str(tmp_path)

    tempfile.tempdir = str(tmp_path)

    if patch_named_tempfile and os.name == "nt":
        _orig = tempfile.NamedTemporaryFile

        def _ntf(*args, **kwargs):
            kwargs.setdefault("delete", False)
            kwargs.setdefault("dir", str(tmp_path))
            return _orig(*args, **kwargs)

        tempfile.NamedTemporaryFile = _ntf

    if add_safe_globals:
        import numpy as np
        import torch.serialization
        import _codecs

        torch.serialization.add_safe_globals([
            np.core.multiarray.scalar,
            np.dtype,
            np.dtypes.Float64DType,
            _codecs.encode,
            np.dtypes.ObjectDType,
            np.ndarray,
            np.core.multiarray._reconstruct,
        ])
