#!/usr/bin/env python3
"""Final packet acceptance over preserved, hashed execution evidence (no model rerun)."""
import hashlib
import json
from pathlib import Path

packet = Path(__file__).resolve().parent
evidence = packet / 'evidence'
review = (packet / 'review.md').read_text()
assert 'Verdict: PASS' in review, 'Independent review has not passed'
for item in json.loads((evidence / 'artifacts.json').read_text()):
    path = Path(item['path'])
    assert path.stat().st_size == item['bytes'], f'Artifact size changed: {path}'
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    assert digest.hexdigest() == item['sha256'], f'Artifact hash changed: {path}'
for filename, digest in json.loads((evidence / 'source-files.json').read_text()).items():
    path = packet.parents[1] / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'Source changed: {path}'
strict_passes = 0
for model in ('ecdet_s', 'ecseg_s', 'ecpose_s', 'rfdetr_det_small', 'rfdetr_seg_medium'):
    for device in ('cpu', 'cuda', 'opencl'):
        raw = json.loads((evidence / f'{model}-{device}_metrics.json').read_text())
        task = json.loads((evidence / f'{model}-task_{device}.json').read_text())
        assert raw['process_exit'] == 0, (model, device)
        assert task['task_equivalence_pass'], (model, device)
        assert task['reference_kept'] > 0 and task['selection_equal']
        assert not any(task['kept_mismatch'].values()), (model, device)
        strict_passes += int(raw['strict_pass'])
assert strict_passes == 7, 'Raw-pass count changed; deviations must stay explicit'
for device, count in (('cpu', 14), ('cuda', 27), ('opencl', 32), ('hip', 3)):
    log = (evidence / f'{device}-ctest.log').read_text()
    full = (evidence / f'{device}-LastTest.log').read_text()
    assert f'0 tests failed out of {count}' in log, device
    assert '[  SKIPPED ]' not in full and '[  FAILED  ]' not in full, device
assert 'Format check passed.' in (evidence / 'format.log').read_text()
assert '290 entries' in (evidence / 'opset-table.log').read_text()
print(json.dumps({'packet_accepted': True, 'local_model_legs': 15,
                  'strict_raw_passes': 7, 'approved_task_deviation_legs': 8,
                  'available_catalog_entries': 29, 'deferred_placeholder': 1,
                  'amd_cloud_testing': 'excluded; release requirement still open',
                  'release_published': False}, indent=2))
