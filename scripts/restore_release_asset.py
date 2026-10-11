import argparse, hashlib, shutil
from pathlib import Path
from zipfile import ZipFile

parser=argparse.ArgumentParser()
parser.add_argument('archive',type=Path)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
with ZipFile(args.archive) as archive:
    for info in archive.infolist():
        if info.is_dir(): continue
        target=(root/info.filename).resolve()
        if not target.is_relative_to(root) or target==root: raise ValueError('Unsafe ZIP member')
        if target.exists():
            with archive.open(info) as source:
                h=hashlib.sha256()
                for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
            if hashlib.sha256(target.read_bytes()).hexdigest()!=h.hexdigest():
                raise FileExistsError('Refusing to overwrite different existing file: '+str(target))
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        with archive.open(info) as source,target.open('xb') as destination:shutil.copyfileobj(source,destination)
# Expand the one large source-table CSV compressed for GitHub web uploads.
for path in root.glob('analysis/upgrade/tables/upgrade_v2/*.csv.gz'):
    target=path.with_suffix('')
    if not target.exists():
        import gzip
        with gzip.open(path,'rb') as source,target.open('xb') as destination:shutil.copyfileobj(source,destination)
print('Release asset restored without replacing different existing files')
