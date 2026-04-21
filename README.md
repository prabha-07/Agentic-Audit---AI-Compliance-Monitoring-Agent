# Agentic-AI-Compliance-Monitoring-System

## macOS (Apple Silicon / Intel): install native deps before `pip install`

`xhtml2pdf` pulls in `svglib` → `rlpycairo` → **`pycairo`**, which links against **Cairo**. If `pip` tries to compile `pycairo` and fails with `pkg-config` / `cairo` not found, install:

```bash
brew install cairo pkg-config
```

Then activate your venv and run `pip install -r requirements.txt` again.

If `brew` is not installed, see [https://brew.sh](https://brew.sh).