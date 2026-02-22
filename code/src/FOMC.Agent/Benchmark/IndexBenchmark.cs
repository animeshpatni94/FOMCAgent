using System.Diagnostics;
using FOMC.Data.Services;
using Microsoft.Extensions.Logging;

namespace FOMC.Agent.Benchmark;

/// <summary>
/// HNSW ef-search recall / latency benchmark against exact (brute-force) search.
///
/// What it measures:
///   For a set of random query vectors sampled from the collection, run:
///     1. Exact search   — ground-truth top-k (Qdrant exact=true mode)
///     2. HNSW search    — at increasing ef values (32, 64, 128, 256)
///   Then compute:
///     - Recall@k = |HNSW ∩ Exact| / k    (fraction of true top-k that were found)
///     - Mean query latency (ms)
///
/// Why ef matters (HNSW tuning knob):
///   ef is the beam width during greedy graph traversal at query time.
///   Low ef  → faster but may miss some true neighbours (lower recall)
///   High ef → slower but closer to exact (higher recall)
///   Sweet spot for most RAG workloads: ef=64–128 gives >0.95 recall at <10ms.
///
/// HNSW vs IVF — see README.md for the full comparison.
/// See scripts/benchmark.py for the Python FAISS IVF vs HNSW cross-comparison.
/// </summary>
public class IndexBenchmark
{
    private readonly QdrantService _qdrant;
    private readonly EmbeddingService _embedder;
    private readonly ILogger<IndexBenchmark> _logger;

    private static readonly int[] EfValues = [16, 32, 64, 128, 256];
    private const int TopK      = 10;
    private const int NumTrials = 20;

    public IndexBenchmark(QdrantService qdrant, EmbeddingService embedder,
                          ILogger<IndexBenchmark> logger)
    {
        _qdrant   = qdrant;
        _embedder = embedder;
        _logger   = logger;
    }

    public async Task RunAsync(CancellationToken ct = default)
    {
        // Sample query vectors from real text prompts — more realistic than random noise
        var queryTexts = new[]
        {
            "federal funds rate target range",
            "inflation outlook consumer prices",
            "labor market employment conditions",
            "balance sheet asset purchases quantitative easing",
            "financial stability risks banking sector",
            "economic growth GDP projections",
            "monetary policy tightening interest rates",
            "committee vote unanimous decision",
            "supply chain disruptions goods prices",
            "wage growth nominal earnings",
            "housing market mortgage rates",
            "global economic conditions international",
            "bank reserves monetary base",
            "unemployment rate job creation",
            "core PCE inflation measure",
            "forward guidance dot plot projections",
            "neutral interest rate equilibrium",
            "risk management uncertainty outlook",
            "credit conditions financial markets",
            "soft landing economic scenario"
        };

        // Limit to NumTrials
        var texts = queryTexts.Take(NumTrials).ToArray();

        Console.WriteLine($"Benchmark: {texts.Length} queries, top-{TopK}\n");
        Console.WriteLine($"{"ef",-8} {"Recall@" + TopK,-14} {"Avg Latency (ms)",-20} {"Min (ms)",-12} {"Max (ms)",-10}");
        Console.WriteLine(new string('─', 70));

        // Embed all queries upfront
        var queryVectors = await _embedder.GetEmbeddingsAsync(texts, ct);

        // Ground truth: exact search for each query
        _logger.LogInformation("Computing exact search ground truth…");
        var groundTruth = new List<HashSet<string>>();
        for (int i = 0; i < queryVectors.Length; i++)
        {
            var exact = await _qdrant.SearchAsync(queryVectors[i], TopK,
                exact: true, ct: ct);
            groundTruth.Add(exact.Select(r => r.Chunk.ChunkId).ToHashSet());
        }

        // HNSW at each ef value
        foreach (var ef in EfValues)
        {
            var latencies  = new List<double>();
            int totalHits  = 0;
            int totalPossible = TopK * texts.Length;

            for (int i = 0; i < queryVectors.Length; i++)
            {
                var sw = Stopwatch.StartNew();
                var results = await _qdrant.SearchAsync(queryVectors[i], TopK,
                    hnswEf: ef, ct: ct);
                sw.Stop();

                latencies.Add(sw.Elapsed.TotalMilliseconds);

                var found = results.Select(r => r.Chunk.ChunkId).ToHashSet();
                totalHits += found.Intersect(groundTruth[i]).Count();
            }

            double recall = (double)totalHits / totalPossible;
            double avgMs  = latencies.Average();
            double minMs  = latencies.Min();
            double maxMs  = latencies.Max();

            Console.WriteLine($"{ef,-8} {recall:P2,-14} {avgMs:F2,-20} {minMs:F2,-12} {maxMs:F2}");
        }

        Console.WriteLine(new string('─', 70));
        Console.WriteLine("""

        Interpretation:
          - Recall@10 = fraction of true top-10 neighbours found by HNSW
          - ef=64 typically achieves >0.95 recall with single-digit ms latency
          - Diminishing returns above ef=128 for most RAG workloads
          - See scripts/benchmark.py for FAISS IVF vs HNSW cross-comparison

        HNSW build params (set at index creation):
          m=16            graph connectivity (higher = better recall, more RAM)
          ef_construct=100  build-time beam width (higher = better graph quality)
        """);
    }
}
