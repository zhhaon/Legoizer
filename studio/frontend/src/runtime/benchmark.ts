import type { BenchmarkIndex, OfficialEvaluation } from '@/types/benchmark';

interface StaticBenchmark {
  schema_version: number;
  index: BenchmarkIndex;
  details: Record<string, { evaluation: OfficialEvaluation; source: string }>;
}
let pending: Promise<StaticBenchmark> | undefined;
function load(): Promise<StaticBenchmark> {
  return pending ??= fetch(`${import.meta.env.BASE_URL}benchmark.json`).then(async response => {
    if (!response.ok) throw new Error(`Benchmark snapshot: HTTP ${response.status}`);
    const data: StaticBenchmark = await response.json();
    if (data.schema_version !== 1 || data.index.total !== data.index.evaluations.length
      || data.index.evaluations.some(row => !data.details[row.signature])) {
      throw new Error('Invalid Benchmark snapshot');
    }
    return data;
  }).catch(error => { pending = undefined; throw error; });
}
export async function staticBenchmark() { return (await load()).index; }
export async function staticBenchmarkDetail(signature: string) {
  const result = (await load()).details[signature];
  if (!result || result.evaluation.signature !== signature) throw new Error('Unknown Benchmark signature');
  return result;
}
