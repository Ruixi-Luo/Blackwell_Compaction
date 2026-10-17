import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from blackwell.cache_reuse import import_api_cache
from blackwell.storage import ROOT, fingerprint, read_json, write_json


class CacheReuseTests(unittest.TestCase):
    def test_import_preserves_request_identity_and_refuses_changed_reader(self):
        config = read_json(ROOT / 'configs/local_comparison.json')
        old = read_json(ROOT / 'configs/pilot.json')
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as temporary:
            root = Path(temporary)
            source = root / 'runs' / old['run_name']
            destination = root / 'runs' / config['run_name']
            write_json(source / 'manifest.json', {'status':'complete','identity':{'config':old}})
            request = {'base_url':old['base_url'], 'payload':{'model':old['model'], 'messages':[]}}
            key = fingerprint(request)
            value = {'request':request, 'response':{'choices':[]}}
            write_json(source / 'api_cache' / f'{key}.json', value)
            with patch('blackwell.cache_reuse.ROOT', root):
                with self.assertRaisesRegex(ValueError, 'configuration differs'):
                    import_api_cache({**config, 'temperature':1}, destination)
                self.assertFalse((destination / 'cache_import.json').exists())
                import_api_cache(config, destination)
                self.assertEqual(read_json(destination / 'api_cache' / f'{key}.json'), value)
                import_api_cache(config, destination, offline=True)
                write_json(destination / 'api_cache' / f'{key}.json', {**value, 'response':{}})
                with self.assertRaisesRegex(ValueError, 'response changed'):
                    import_api_cache(config, destination, offline=True)


if __name__ == '__main__':
    unittest.main()
