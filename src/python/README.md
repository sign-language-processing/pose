# Pose Format

## Publishing
```bash
pip install --upgrade build twine pyopenssl cryptography requests-toolbelt
rm -rf dist
python3 -m build
python3 -m twine upload dist/*
```