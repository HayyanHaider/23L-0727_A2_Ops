"""Create the credential-free, self-contained serving and evidence submission."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'submission/23L-0727_A2_Submission.zip'


def main():
    verification = json.loads((ROOT / 'submission/evidence/full_assignment/evidence/FINAL_VERIFICATION.json').read_text())
    assert verification['all_required_tasks_verified']
    files = set()
    for folder in ['visionguard', 'scripts', 'tests', 'docs', 'notebooks', 'postman', 'samples', '.github', '.devcontainer', 'submission/evidence']:
        files.update(p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for name in ['README.md', 'Dockerfile', 'Dockerfile.baseline', 'compose.yaml', 'wsgi.py', 'pyproject.toml', '.env.example', '.gitignore', '.dockerignore', '.dvcignore', '.dvc/config', '.dvc/.gitignore', 'data/processed.dvc', 'data/splits.dvc', 'data/.gitignore', 'models/champion.keras', 'models/champion.json', 'submission/23L-0727_VisionGuard.pdf', 'submission/README.txt']:
        if (ROOT / name).is_file(): files.add(ROOT / name)
    files.update(ROOT.glob('requirements*.txt'))
    for p in files:
        assert p.name not in ['.env', 'kaggle.json'] and '.git' not in p.parts and p.suffix not in ['.db', '.sqlite3', '.zip']
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    DEST.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(DEST, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(files): z.write(p, '23L-0727_A2_Ops/' + str(p.relative_to(ROOT)))
        z.writestr('23L-0727_A2_Ops/FILE_SHA256.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(DEST) as z:
        assert z.testzip() is None
        assert z.read('23L-0727_A2_Ops/models/champion.keras') == (ROOT / 'models/champion.keras').read_bytes()
        assert z.read('23L-0727_A2_Ops/submission/23L-0727_VisionGuard.pdf') == (ROOT / 'submission/23L-0727_VisionGuard.pdf').read_bytes()
    print(json.dumps(dict(archive=str(DEST), files=len(files)+1, bytes=DEST.stat().st_size, crc_verified=True, model_and_pdf_match=True), indent=2))


if __name__ == '__main__':
    main()
