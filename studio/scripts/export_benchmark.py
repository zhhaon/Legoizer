"""Capture real official Benchmark results, including configuration details.

The requested snapshot label is stored without a capture timestamp. Refresh explicitly;
ordinary Pages builds never need the upstream service.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from studio.backend.benchmark import evaluations, evaluation_detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--display-date', default='2026-11-16')
    parser.add_argument('--index-file', type=Path)
    args = parser.parse_args()
    index = json.loads(args.index_file.read_text()) if args.index_file else evaluations()
    details = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        tasks = {pool.submit(evaluation_detail, row['signature']): row['signature'] for row in index['evaluations']}
        for future in as_completed(tasks):
            signature = tasks[future]
            result = future.result()
            if result['evaluation']['signature'] != signature:
                raise ValueError('Detail signature mismatch')
            if 'configuration' not in result['evaluation']:
                raise ValueError(f'Missing configuration for {signature}')
            details[signature] = result
            if len(details) % 100 == 0:
                print(f'Captured {len(details)}/{index["total"]} details', flush=True)
    assert len(details) == index['total'] == len(index['evaluations'])
    index.pop('fetched_at', None)
    index.pop('captured_at', None)
    index.update(snapshot_date=args.display_date, cache_seconds=0)
    payload = {'schema_version': 1, 'index': index, 'details': details}
    target = Path(__file__).resolve().parents[1] / 'frontend/public/benchmark.json'
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(encoded + '\n')
    temporary.replace(target)
    print(f'Wrote {target}: {len(encoded)} bytes', flush=True)


if __name__ == '__main__':
    main()
