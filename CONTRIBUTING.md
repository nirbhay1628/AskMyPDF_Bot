# Contributing

Thanks for considering contributing to this project.

## How to contribute

1. Fork the repository.
2. Create a feature branch from `main`.
3. Make your changes.
4. Run and test the bot locally.
5. Open a pull request.

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m bot.main
```

## Guidelines

- Keep changes focused and minimal.
- Do not commit secrets or `.env` files.
- Update `README.md` when behavior or setup changes.
- Prefer clear, production-safe code.

## Reporting issues

When filing an issue, include:
- what you expected
- what happened
- any error message
- steps to reproduce
- relevant environment details
