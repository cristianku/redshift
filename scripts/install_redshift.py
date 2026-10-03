#!/usr/bin/env python3
"""Install Redshift, its pinned Hugging Face model, and a systemd service."""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import grp
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import platform
import pwd
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path('/opt/redshift')
UNIT = Path('/etc/systemd/system/redshift.service')
MARKER = '.redshift-install.json'
UNIT_MARKER = '# Managed by Redshift installer.'
SERVICE = 'redshift.service'
MODEL_ID = 'redshift-qwen3.8-27b'
MODEL_SHA256 = '31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34'
DEFAULT_MODEL = Path('/opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf')
MODEL_REVISION = '97c30c65c8d9a3e73f9fdfb50f1d1a669e9a2827'
MODEL_URL = ('https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/'
             + MODEL_REVISION + '/' + DEFAULT_MODEL.name)
MODEL_BYTES = 18973870432
CUDA = Path('/usr/local/cuda-12.9')
SOURCE = Path(__file__).resolve().parents[1]


@dataclass
class Options:
    model: Path
    model_url: str | None
    model_sha256: str
    host: str
    public_host: str
    port: int
    context: int
    dry_run: bool
    offline: bool
    reuse_python: Path | None


def parse_options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, help='existing GGUF, or destination for an explicit download')
    parser.add_argument('--model-url', help='override the pinned default Hugging Face download URL')
    parser.add_argument('--model-sha256', default=MODEL_SHA256, help='expected artifact SHA-256')
    parser.add_argument('--host', default='127.0.0.1', help='IPv4 listen address')
    parser.add_argument('--public-host', help='host/IP used in generated Copilot configuration')
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--context', type=int, default=139264)
    parser.add_argument('--dry-run', action='store_true', help='print the plan without writes or network access')
    parser.add_argument('--offline', action='store_true', help='forbid package/model downloads; requires existing CUDA and --reuse-python')
    parser.add_argument('--reuse-python', type=Path, help='existing compatible venv Python, used instead of downloading Python dependencies')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not 32 <= args.context <= 139264:
        raise ValueError('port must be 1..65535 and context 32..139264')
    try:
        ipaddress.IPv4Address(args.host)
    except ipaddress.AddressValueError as error:
        raise ValueError('--host must be an IPv4 address') from error
    if not re.fullmatch(r'[0-9a-fA-F]{64}', args.model_sha256):
        raise ValueError('--model-sha256 must contain 64 hexadecimal characters')
    if args.model_url:
        url = urllib.parse.urlparse(args.model_url)
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password:
            raise ValueError('--model-url must be an HTTP(S) URL without embedded credentials')
    public = args.public_host or args.host
    if public == '0.0.0.0':
        raise ValueError('--public-host is required with --host 0.0.0.0')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', public):
        raise ValueError('--public-host must be a hostname or IPv4 address')
    url = args.model_url
    if args.model is not None:
        model = args.model
    elif url:
        model = ROOT / 'models' / DEFAULT_MODEL.name
    elif DEFAULT_MODEL.is_file():
        model = DEFAULT_MODEL
    else:
        model = ROOT / 'models' / DEFAULT_MODEL.name
        url = MODEL_URL
    model = model.expanduser().absolute()
    if any(ord(c) < 32 for c in str(model)):
        raise ValueError('model path contains control characters')
    if args.offline and args.reuse_python is None:
        raise ValueError('--offline requires --reuse-python pointing to an existing venv interpreter')
    if args.offline:
        url = None
    return Options(model, url, args.model_sha256.lower(), args.host,
                   public, args.port, args.context, args.dry_run, args.offline, args.reuse_python)


def run(*command, **kwargs):
    print('+ ' + ' '.join(str(c) for c in command), flush=True)
    return subprocess.run([str(c) for c in command], check=True, **kwargs)


def output(*command):
    return subprocess.check_output([str(c) for c in command], text=True).strip()


def check_ownership(root, unit):
    if root.is_symlink() or unit.is_symlink():
        raise ValueError('refusing symlink installation root or unit')
    if root.exists():
        try:
            marker = json.loads((root / MARKER).read_text())
            if marker.get('format') != 1:
                raise ValueError('unknown marker')
        except (OSError, ValueError, AttributeError) as error:
            raise ValueError(f'unmanaged installation directory: {root}') from error
    if unit.exists() and (UNIT_MARKER not in unit.read_text()
                          or f'WorkingDirectory={root}/current\n' not in unit.read_text()):
        raise ValueError(f'unmanaged service file: {unit}')
    current = root / 'current'
    if current.exists() and not current.is_symlink():
        raise ValueError('unmanaged current release directory')
    if current.is_symlink() and not current.resolve().is_relative_to(root.resolve()):
        raise ValueError('current release points outside the installation')


def check_loaded_unit(unit, query=output):
    fragment = query('systemctl', 'show', SERVICE, '-p', 'FragmentPath', '--value')
    dropins = query('systemctl', 'show', SERVICE, '-p', 'DropInPaths', '--value')
    if fragment and fragment != str(unit):
        raise ValueError(f'unmanaged loaded service: {fragment}')
    if dropins:
        raise ValueError('existing redshift service has unmanaged drop-in configuration')


def validate_model_visibility(model):
    actual = model.resolve()
    hidden = ('/home', '/root', '/run/user', '/tmp', '/var/tmp')
    if any(model.is_relative_to(path) or actual.is_relative_to(path) for path in map(Path, hidden)):
        raise ValueError('model is hidden by service ProtectHome/PrivateTmp; use /opt/models or the automatic download')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch_model(url, target):
    with urllib.request.urlopen(url, timeout=60) as response, target.open('wb') as file:
        shutil.copyfileobj(response, file, length=1024 * 1024)
        file.flush()
        os.fsync(file.fileno())


def ensure_model(path, url, expected, fetch=fetch_model):
    if path.is_symlink():
        raise ValueError('model symlink is not supported; pass its actual path')
    if path.exists():
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f'model SHA-256 mismatch; existing file preserved: {path}')
        print(f'Reusing verified model: {path}', flush=True)
        return
    if not url:
        raise ValueError(f'missing model {path}; pass --model FILE or --model-url URL')
    path.parent.mkdir(parents=True, exist_ok=True)
    if url == MODEL_URL and shutil.disk_usage(path.parent).free < MODEL_BYTES + 1024**3:
        raise ValueError('at least 20 GB of free space is required for the default model download')
    fd, name = tempfile.mkstemp(prefix='.redshift-model-', dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        print(f'Downloading model to {path}; verifying SHA-256 before publication.', flush=True)
        fetch(url, temporary)
        if sha256(temporary) != expected:
            raise ValueError('downloaded model SHA-256 mismatch')
        temporary.chmod(0o644)
        # Hard-link publication refuses to replace a file created concurrently.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def quote_unit(value):
    text = str(value)
    if any(ord(c) < 32 for c in text):
        raise ValueError('control character in systemd argument')
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%').replace('$', '$$') + '"'


def render_unit(root, model, host, port, context, groups):
    current = root / 'current'
    group_line = 'SupplementaryGroups=' + ' '.join(groups) + '\n' if groups else ''
    return (f'{UNIT_MARKER}\n[Unit]\nDescription=Redshift Qwen GPU endpoint\n'
            'After=network-online.target\nWants=network-online.target\n'
            'StartLimitIntervalSec=120\nStartLimitBurst=3\n\n[Service]\n'
            f'Type=simple\nUser=redshift\nGroup=redshift\n{group_line}'
            f'WorkingDirectory={current}\n'
            f'ExecStart={quote_unit(current / ".venv/bin/python")} -m qvelox.server '
            f'{quote_unit(model)} --host {host} --port {port} --context {context}\n'
            'Environment=PYTHONDONTWRITEBYTECODE=1\nEnvironment=CUDA_CACHE_DISABLE=1\n'
            'Restart=on-failure\nRestartSec=5\nTimeoutStopSec=30\n'
            'NoNewPrivileges=true\nProtectSystem=strict\nProtectHome=true\nPrivateTmp=true\n'
            '\n[Install]\nWantedBy=multi-user.target\n')


def atomic_text(path, text):
    fd, temporary = tempfile.mkstemp(prefix='.redshift-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def switch_release(root, target):
    temporary = root / ('.current-' + uuid.uuid4().hex)
    temporary.symlink_to(target)
    try:
        os.replace(temporary, root / 'current')
    finally:
        temporary.unlink(missing_ok=True)


def activate(root, unit_path, release, unit_text, runner, check, *, was_active, was_enabled):
    current = root / 'current'
    previous = os.readlink(current) if current.is_symlink() else None
    old_unit = unit_path.read_text() if unit_path.exists() else None
    try:
        atomic_text(unit_path, unit_text)
        switch_release(root, release)
        runner('systemctl', 'daemon-reload')
        runner('systemctl', 'enable', SERVICE)
        runner('systemctl', 'restart', SERVICE)
        check()
    except BaseException:
        runner('systemctl', 'stop', SERVICE)
        if not was_enabled:
            runner('systemctl', 'disable', SERVICE)
        if old_unit is None:
            unit_path.unlink(missing_ok=True)
        else:
            atomic_text(unit_path, old_unit)
        if previous is None:
            current.unlink(missing_ok=True)
        else:
            switch_release(root, previous)
        runner('systemctl', 'daemon-reload')
        if was_active and old_unit is not None:
            runner('systemctl', 'start', SERVICE)
        raise


def copilot_config(options):
    maximum = min(16384, options.context // 8)
    return [{'name': 'Redshift V100', 'vendor': 'customendpoint', 'apiType': 'chat-completions',
             'models': [{'id': MODEL_ID, 'name': 'Redshift Qwen3.8 27B (V100)',
                         'url': f'http://{options.public_host}:{options.port}/v1/chat/completions',
                         'toolCalling': True, 'vision': False, 'thinking': True,
                         'supportsReasoningEffort': ['low', 'medium', 'xhigh'],
                         'reasoningEffortFormat': 'chat-completions',
                         'contextWindow': options.context,
                         'maxInputTokens': options.context - maximum,
                         'maxOutputTokens': maximum, 'modelOptions': {'temperature': 0}}]}]


def configure_packages(options):
    if options.offline:
        if not all(shutil.which(name) for name in ('c++', 'make', 'python3')) or not (CUDA / 'bin/nvcc').exists():
            raise ValueError('offline installation requires existing compiler, make, Python and CUDA 12.9')
        return
    required = ['build-essential', 'python3', 'python3-venv', 'curl', 'ca-certificates']
    missing = []
    for package in required:
        result = subprocess.run(['dpkg-query', '-W', '-f=${db:Status-Status}', package],
                                capture_output=True, text=True)
        if result.returncode or result.stdout != 'installed':
            missing.append(package)
    if missing:
        run('apt-get', 'update')
        run('apt-get', 'install', '-y', *missing,
            env=dict(os.environ, DEBIAN_FRONTEND='noninteractive'))
    if not (CUDA / 'bin/nvcc').exists():
        with tempfile.TemporaryDirectory(prefix='redshift-cuda-keyring-') as folder:
            package = Path(folder) / 'cuda-keyring.deb'
            fetch_model('https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb', package)
            run('dpkg', '-i', package)
        run('apt-get', 'update')
        run('apt-get', 'install', '-y', 'cuda-toolkit-12-9',
            env=dict(os.environ, DEBIAN_FRONTEND='noninteractive'))
    version = output(CUDA / 'bin/nvcc', '--version')
    if 'release 12.9,' not in version:
        raise ValueError('CUDA Toolkit 12.9 is required for this Volta build')


def ensure_gpu(offline=False):
    result = subprocess.run(['nvidia-smi'], capture_output=True) if shutil.which('nvidia-smi') else None
    if result is None or result.returncode:
        if offline:
            raise ValueError('GPU driver unavailable; offline mode cannot install it')
        container = subprocess.run(['systemd-detect-virt', '--container'], capture_output=True)
        if container.returncode == 0:
            raise ValueError('GPU unavailable in container: configure NVIDIA access on its host first')
        run('apt-get', 'update')
        run('apt-get', 'install', '-y', 'linux-headers-' + platform.release(), 'nvidia-driver-580',
            env=dict(os.environ, DEBIAN_FRONTEND='noninteractive'))
        raise RuntimeError('NVIDIA driver installed. Reboot this new server yourself, then rerun the same command.')
    rows = output('nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits').splitlines()
    if len(rows) != 1:
        raise ValueError('this installer expects one visible NVIDIA GPU')
    name, memory, driver = [value.strip() for value in rows[0].split(',')]
    if 'V100' not in name or int(memory) < 30000:
        raise ValueError('this build requires a Tesla V100 32 GB')
    if int(driver.split('.')[0]) < 575:
        raise ValueError('existing NVIDIA driver is too old for CUDA 12.9; update it separately')
    print(f'Using existing GPU and driver: {name}, {driver}', flush=True)


def health_and_probe(options):
    host = '127.0.0.1' if options.host == '0.0.0.0' else options.host
    base = f'http://{host}:{options.port}'
    deadline = time.monotonic() + 180
    while True:
        try:
            with urllib.request.urlopen(base + '/health', timeout=5) as response:
                health = json.load(response)
            if health.get('engine') != 'redshift' or not health.get('ready') or health.get('context') != options.context:
                raise RuntimeError('unexpected health response')
            break
        except (OSError, ValueError, RuntimeError):
            if time.monotonic() >= deadline:
                raise RuntimeError('Redshift did not become healthy within 180 seconds')
            time.sleep(1)
    payload = {'model': MODEL_ID, 'messages': [{'role': 'user', 'content': 'Quanto fa 6 per 7? Rispondi solo con il numero.'}],
               'temperature': 0, 'max_tokens': min(16, options.context - 29)}
    request = urllib.request.Request(base + '/v1/chat/completions', json.dumps(payload).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=45) as response:
        answer = json.load(response)
    if answer['choices'][0]['message']['content'].strip() != '42':
        raise RuntimeError('real model inference probe failed')


def install(options):
    check_ownership(ROOT, UNIT)
    validate_model_visibility(options.model)
    if not options.model.exists() and not options.model_url:
        raise ValueError(f'missing model {options.model}; provide a file or an explicit download URL')
    if options.model.is_symlink():
        raise ValueError('pass the actual model path, not a symlink')
    if os.geteuid() != 0:
        raise ValueError('run scripts/install-redshift.sh with sudo')
    release_info = platform.freedesktop_os_release()
    if release_info.get('ID') != 'ubuntu' or release_info.get('VERSION_ID') != '24.04' or platform.machine() != 'x86_64':
        raise ValueError('this installer supports Ubuntu 24.04 x86_64')
    if not Path('/run/systemd/system').is_dir():
        raise ValueError('systemd must be running')
    check_loaded_unit(UNIT)
    was_active = subprocess.run(['systemctl', 'is-active', '--quiet', SERVICE]).returncode == 0
    was_enabled = subprocess.run(['systemctl', 'is-enabled', '--quiet', SERVICE]).returncode == 0
    if not was_active:
        with socket.socket() as connection:
            connection.bind((options.host, options.port))
    ensure_gpu(options.offline)
    configure_packages(options)
    ROOT.mkdir(mode=0o755, exist_ok=True)
    atomic_text(ROOT / MARKER, json.dumps({'format': 1, 'installer': 'redshift'}) + '\n')
    ensure_model(options.model, options.model_url, options.model_sha256)
    try:
        user = pwd.getpwnam('redshift')
        if user.pw_uid == 0 or user.pw_shell not in ('/usr/sbin/nologin', '/bin/false'):
            raise ValueError('existing redshift account is not a dedicated service user')
    except KeyError:
        run('useradd', '--system', '--user-group', '--home-dir', '/nonexistent',
            '--shell', '/usr/sbin/nologin', 'redshift')
    groups = [group for group in ('video', 'render') if any(g.gr_name == group for g in grp.getgrall())]
    run('runuser', '-u', 'redshift', '--', 'python3', '-c',
        'import sys; open(sys.argv[1], "rb").read(1)', options.model)
    releases = ROOT / 'releases'
    releases.mkdir(exist_ok=True)
    release = releases / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8])
    release.mkdir(mode=0o755)
    for name in ('src', 'qvelox', 'tests', 'scripts', 'tools'):
        shutil.copytree(SOURCE / name, release / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('Makefile', 'requirements-server.txt', 'LICENSE', 'README.md', 'THIRD_PARTY.md'):
        shutil.copy2(SOURCE / name, release / name)
    jobs = min(os.cpu_count() or 1, 20)
    run('make', '-j' + str(jobs), 'JOBS=' + str(jobs), 'CUDA_DIR=' + str(CUDA), cwd=release)
    run('python3', '-m', 'venv', *(('--without-pip',) if options.reuse_python else ()), release / '.venv')
    python = release / '.venv/bin/python'
    if options.reuse_python:
        details = json.loads(output(options.reuse_python, '-c',
                                   'import sys,sysconfig,json; print(json.dumps({"version":list(sys.version_info[:2]),'
                                   '"site":sysconfig.get_path("purelib")}))'))
        if details['version'] != list(os.sys.version_info[:2]):
            raise ValueError('reused Python environment must match the server Python version')
        target = Path(output(python, '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'))
        shutil.copytree(details['site'], target, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        run(python, '-c', 'import importlib.metadata as m; '
            'assert m.version("tokenizers")=="0.22.2"; assert m.version("Jinja2")=="3.1.6"')
    else:
        run(python, '-m', 'pip', 'install', '-r', release / 'requirements-server.txt')
    run(python, '-c', 'import sys; from qvelox.gguf import read_gguf; from qvelox.model import validate_qwen27b; '
        'validate_qwen27b(read_gguf(sys.argv[1])); print("GGUF schema validated")', options.model, cwd=release)
    # Test the isolated build. An active previous instance must retain its GPU.
    env = dict(os.environ, QVELOX_CUDA='0' if was_active or options.offline else '1',
               QVELOX_MODEL='' if was_active or options.offline else str(options.model))
    run(python, '-m', 'unittest', 'discover', '-s', 'tests', '-v', cwd=release, env=env)
    unit = render_unit(ROOT, options.model, options.host, options.port, options.context, groups)
    unit_check = release / SERVICE
    unit_check.write_text(unit.replace(str(ROOT / 'current'), str(release)))
    run('systemd-analyze', 'verify', unit_check)
    activate(ROOT, UNIT, release, unit, run, lambda: health_and_probe(options),
             was_active=was_active, was_enabled=was_enabled)
    configuration = ROOT / 'copilot-model.json'
    atomic_text(configuration, json.dumps(copilot_config(options), indent=2) + '\n')
    manifest = {'format': 1, 'model': str(options.model), 'model_sha256': options.model_sha256,
                'host': options.host, 'port': options.port, 'context': options.context,
                'release': str(release), 'source_sha256': sha256(release / 'qvelox/server.py')}
    atomic_text(ROOT / MARKER, json.dumps(manifest, indent=2) + '\n')
    print(f'Redshift is healthy: http://{options.public_host}:{options.port}/v1/chat/completions', flush=True)
    print(f'Copilot configuration: {configuration}', flush=True)


def main(argv=None):
    try:
        options = parse_options(argv)
        if options.dry_run:
            print(json.dumps({'target': 'Ubuntu 24.04 x86_64 / Tesla V100 32GB',
                              'install_root': str(ROOT), 'service': SERVICE, 'cuda': str(CUDA),
                              'model': str(options.model), 'download_model': bool(options.model_url),
                              'model_sha256': options.model_sha256,
                              'copilot': copilot_config(options)}, indent=2))
            return 0
        install(options)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'Installation failed: {error}', file=os.sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
