"""Installer safeguards: never mutate unrelated installations or lose rollback."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/install_redshift.py'
spec = importlib.util.spec_from_file_location('redshift_installer', SCRIPT)
installer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = installer
spec.loader.exec_module(installer)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / 'redshift'
        self.unit = self.base / 'redshift.service'

    def test_refuses_unmanaged_directory_and_service(self):
        self.root.mkdir()
        (self.root / 'important.txt').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'unmanaged'):
            installer.check_ownership(self.root, self.unit)
        self.assertEqual((self.root / 'important.txt').read_text(), 'keep')
        (self.root / installer.MARKER).write_text('{"format":1}')
        self.unit.write_text('[Service]\nExecStart=/usr/bin/other\n')
        with self.assertRaisesRegex(ValueError, 'unmanaged'):
            installer.check_ownership(self.root, self.unit)
        self.assertIn('/usr/bin/other', self.unit.read_text())

    def test_bad_download_never_publishes_model(self):
        model = self.base / 'model.gguf'
        def fetch(url, target):
            target.write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            installer.ensure_model(model, 'https://example.test/model', '0' * 64, fetch)
        self.assertFalse(model.exists())
        self.assertEqual(list(self.base.iterdir()), [])

    def test_reuses_existing_model_and_preserves_bad_existing_file(self):
        model = self.base / 'model.gguf'
        model.write_bytes(b'model')
        expected = hashlib.sha256(b'model').hexdigest()
        def no_download(*args):
            self.fail('existing model must not be downloaded or overwritten')
        installer.ensure_model(model, 'https://example.test/model', expected, no_download)
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            installer.ensure_model(model, None, '0' * 64, no_download)
        self.assertEqual(model.read_bytes(), b'model')

    def test_download_is_published_only_after_hash_matches(self):
        model = self.base / 'model.gguf'
        def fetch(url, target):
            self.assertFalse(model.exists())
            target.write_bytes(b'valid')
        installer.ensure_model(model, 'https://example.test/model', hashlib.sha256(b'valid').hexdigest(), fetch)
        self.assertEqual(model.read_bytes(), b'valid')
        self.assertEqual(list(self.base.iterdir()), [model])

    def test_model_symlink_is_rejected(self):
        source = self.base / 'other.gguf'
        source.write_bytes(b'keep')
        target = self.base / 'model.gguf'
        target.symlink_to(source)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            installer.ensure_model(target, None, hashlib.sha256(b'keep').hexdigest())
        self.assertEqual(source.read_bytes(), b'keep')

    def test_unit_escapes_paths_and_includes_gpu_service_user(self):
        unit = installer.render_unit(Path('/opt/redshift'), Path('/models/a "b" % $x.gguf'),
                                     '10.10.10.55', 8081, 32768, ['video'])
        self.assertIn('User=redshift', unit)
        self.assertIn('SupplementaryGroups=video', unit)
        self.assertIn('a \\"b\\" %% $$x.gguf', unit)
        self.assertIn('--host 10.10.10.55 --port 8081 --context 32768', unit)
        with self.assertRaises(ValueError):
            installer.render_unit(self.root, Path('/models/bad\nExecStart=x'), '127.0.0.1', 8081, 32768, [])

    def test_failed_activation_restores_previous_release_and_unit(self):
        self.root.mkdir()
        old, new = self.root / 'old', self.root / 'new'
        old.mkdir(); new.mkdir()
        (self.root / 'current').symlink_to(old)
        previous = '# old owned unit\n'
        self.unit.write_text(previous)
        commands = []
        def run(*args): commands.append(args)
        def check(): raise RuntimeError('failed health')
        with self.assertRaisesRegex(RuntimeError, 'failed health'):
            installer.activate(self.root, self.unit, new, '# new unit\n', run, check,
                               was_active=True, was_enabled=True)
        self.assertEqual((self.root / 'current').resolve(), old)
        self.assertEqual(self.unit.read_text(), previous)
        self.assertIn(('systemctl', 'start', 'redshift.service'), commands)
        self.assertNotIn(('systemctl', 'disable', 'redshift.service'), commands)

    def test_failed_first_activation_removes_only_own_unit_and_link(self):
        self.root.mkdir()
        new = self.root / 'release'; new.mkdir()
        commands = []
        def run(*args): commands.append(args)
        def check(): raise RuntimeError('no GPU')
        with self.assertRaises(RuntimeError):
            installer.activate(self.root, self.unit, new, '# new unit\n', run, check,
                               was_active=False, was_enabled=False)
        self.assertFalse(self.unit.exists())
        self.assertFalse((self.root / 'current').is_symlink())
        self.assertTrue(new.is_dir())
        self.assertIn(('systemctl', 'disable', 'redshift.service'), commands)

    def test_successful_activation_keeps_previous_release(self):
        self.root.mkdir()
        old, new = self.root / 'old', self.root / 'new'
        old.mkdir(); new.mkdir()
        (self.root / 'current').symlink_to(old)
        commands = []
        installer.activate(self.root, self.unit, new, '# good unit\n',
                           lambda *args: commands.append(args), lambda: None,
                           was_active=True, was_enabled=True)
        self.assertEqual((self.root / 'current').resolve(), new)
        self.assertTrue(old.exists())
        self.assertEqual(self.unit.read_text(), '# good unit\n')

    def test_cli_rejects_invalid_target_before_any_mutation(self):
        for arguments in (['--port', '0'], ['--context', '139265'],
                          ['--host', '::1'], ['--model-sha256', 'wrong'],
                          ['--model-url', 'file:///etc/passwd']):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                installer.parse_options(arguments)

    def test_copilot_uses_requested_host_and_bounded_context(self):
        options = installer.parse_options(['--host', '10.10.10.55', '--context', '139264'])
        model = installer.copilot_config(options)[0]['models'][0]
        self.assertEqual(model['url'], 'http://10.10.10.55:8081/v1/chat/completions')
        self.assertTrue(model['toolCalling'])
        self.assertTrue(model.get('thinking'))
        self.assertEqual(model.get('supportsReasoningEffort'), ['low', 'medium', 'xhigh'])
        self.assertEqual(model.get('reasoningEffortFormat'), 'chat-completions')
        self.assertEqual(model['maxInputTokens'], 122880)
        self.assertEqual(model['maxOutputTokens'], 16384)
        self.assertLessEqual(model['maxInputTokens'] + model['maxOutputTokens'], options.context)

    def test_empty_machine_automatically_uses_pinned_huggingface_model(self):
        with patch.object(installer, 'DEFAULT_MODEL', self.base / 'absent.gguf'):
            options = installer.parse_options([])
        self.assertEqual(options.model_url, installer.MODEL_URL)
        self.assertIn(installer.MODEL_REVISION, options.model_url)
        self.assertNotIn('/main/', options.model_url)
        self.assertEqual(options.model_sha256, installer.MODEL_SHA256)

    def test_explicit_local_model_does_not_enable_download(self):
        options = installer.parse_options(['--model', str(self.base / 'offline.gguf')])
        self.assertIsNone(options.model_url)

    def test_offline_cannot_download_any_model(self):
        options = installer.parse_options(['--offline', '--reuse-python', '/venv/bin/python'])
        self.assertIsNone(options.model_url)
        with self.assertRaises(ValueError):
            installer.parse_options(['--offline'])

    def test_dry_run_has_no_install_or_network_actions(self):
        with patch.object(installer, 'install', side_effect=AssertionError('mutation')):
            with redirect_stdout(StringIO()) as printed:
                self.assertEqual(installer.main(['--dry-run']), 0)
        self.assertIn('download_model', printed.getvalue())

    def test_loaded_vendor_unit_or_dropins_are_rejected(self):
        for fragment, dropins in (('/usr/lib/systemd/system/redshift.service', ''),
                                 ('/run/systemd/transient/redshift.service', ''),
                                 (str(self.unit), '/etc/systemd/system/redshift.service.d/local.conf')):
            values = iter((fragment, dropins))
            with self.assertRaisesRegex(ValueError, 'unmanaged'):
                installer.check_loaded_unit(self.unit, lambda *args: next(values))
        values = iter((str(self.unit), ''))
        installer.check_loaded_unit(self.unit, lambda *args: next(values))

    def test_service_hidden_model_paths_fail_before_install(self):
        for path in ('/home/user/model.gguf', '/root/model.gguf', '/run/user/1000/model.gguf',
                     '/tmp/model.gguf', '/var/tmp/model.gguf'):
            with self.assertRaisesRegex(ValueError, 'ProtectHome'):
                installer.validate_model_visibility(Path(path))
        installer.validate_model_visibility(Path('/opt/models/model.gguf'))


if __name__ == '__main__':
    unittest.main()
