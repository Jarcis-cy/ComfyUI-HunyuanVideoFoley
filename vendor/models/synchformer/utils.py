from pathlib import Path

# Note: In the vendorized version we intentionally avoid network downloads.
# If a required file is missing, we raise a clear, actionable error message
# instructing the user to manually place the file.


def check_if_file_exists_else_download(path, fname2link=None, *args, **kwargs):
    """
    Vendorized: Only checks for existence. Does NOT download.
    - If the file exists, return silently.
    - If missing, raise FileNotFoundError with a clear message and (optionally) a known URL mapping.
    """
    p = Path(path)
    if p.exists():
        return
    hint = ""
    if fname2link and isinstance(fname2link, dict):
        url = fname2link.get(p.name)
        if url:
            hint = f" Known URL: {url}"
    raise FileNotFoundError(
        f"Required file not found: {p}. Please download it manually and place it at this path.{hint}"
    )


class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

