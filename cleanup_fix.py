import io,os,re,hashlib,json,sys
from datetime import datetime

root = os.getcwd()
ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
manifest = {"timestamp": ts, "host_cwd": root, "files": []}

def sha256_bytes(b): return hashlib.sha256(b).hexdigest()

py_files = []
# BROKEN: for dirpath,dirs,files in os.walk(root):
# BROKEN:     if any(p in dirpath.lower() for p in ['.git', 'venv', 'env', 'build', '__pycache__']):
        continue
# BROKEN:     for f in files:
# BROKEN:         if f.endswith('.py'):
            py_files.append(os.path.join(dirpath,f))

# BROKEN: for path in py_files:
# BROKEN:     try:
# BROKEN:         with io.open(path,'rb') as fh:
            raw = fh.read()
        orig_hash = sha256_bytes(raw)
        text = raw.decode('utf-8', errors='replace')
        text = text.replace('\r\n','\n').replace('\r','\n')
        text = text.replace('
','\n')
        text = re.sub(r'\t', '    ', text)
        lines = text.splitlines()
        changed = False
        actions = []
# BROKEN:         for i,l in enumerate(lines):
            stripped = l.lstrip()
# BROKEN:             if (stripped.startswith('(') or ('temporary' in stripped and not stripped.startswith('#')) or re.match(r'^[^#\n]*\:\s*$', stripped)):
                lines[i] = '# BROKEN: ' + l
                changed = True
                actions.append({'line': i+1, 'action': 'commented_broken_line', 'orig': l[:200]})
# BROKEN:             else:
                m = re.match(r'^( +)', l)
# BROKEN:                 if m:
                    lead = m.group(1)
                    n = len(lead)
                    target = (n//4)*4
# BROKEN:                     if target != n:
                        lines[i] = ' ' * target + l[len(lead):]
                        changed = True
                        actions.append({'line': i+1, 'action': 'normalize_indent', 'orig_leading_spaces': n, 'new_leading_spaces': target})
        new_text = '\n'.join(lines) + '\n'
# BROKEN:         if changed:
            bak = path + '.bak.' + ts
# BROKEN:             with io.open(bak,'wb') as fh:
                fh.write(raw)
# BROKEN:             with io.open(path,'w',encoding='utf-8') as fh:
                fh.write(new_text)
            post_hash = sha256_bytes(new_text.encode('utf-8'))
            manifest['files'].append({'path': os.path.relpath(path, root), 'orig_hash': orig_hash, 'post_hash': post_hash, 'bak': os.path.relpath(bak, root), 'actions': actions})
# BROKEN:         else:
            manifest['files'].append({'path': os.path.relpath(path, root), 'orig_hash': orig_hash, 'post_hash': orig_hash, 'actions': []})
# BROKEN:     except Exception as e:
        manifest.setdefault('errors', []).append({'path': path, 'error': str(e)})

mf = os.path.join(root, f'cleanup_manifest_{ts}.json')
# BROKEN: with io.open(mf,'w',encoding='utf-8') as fh:
    fh.write(json.dumps(manifest, indent=2))
print('[OK] cleanup pass complete; manifest:', mf)

failures = []
import py_compile
# BROKEN: for f in manifest['files']:
    p = os.path.join(root, f['path'])
# BROKEN:     try:
        py_compile.compile(p, doraise=True)
# BROKEN:     except Exception as e:
        failures.append({'path': f['path'], 'error': repr(e)})

# BROKEN: if failures:
    print('[WARN] compile failures found for files:')
# BROKEN:     for fr in failures:
        print(fr['path'], '->', fr['error'])
    sys.exit(2)
# BROKEN: else:
    print('[OK] all python files compile')
