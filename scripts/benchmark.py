"""
HNSW vs IVF Benchmark
=====================
Compares FAISS HNSWFlat and IVFFlat (IVF) index types on FOMC corpus vectors
exported from Qdrant.

This script complements the in-process C# Qdrant benchmark in
FOMC.Agent/Benchmark/IndexBenchmark.cs which benchmarks ef-search tuning
within Qdrant's HNSW index.  This script provides the FAISS IVF comparison.

Usage:
    pip install qdrant-client faiss-cpu numpy openai python-dotenv
    python scripts/benchmark.py

Prerequisites:
    - Qdrant running on localhost:6333
    - FOMC documents already ingested (run: dotnet run --project src/FOMC.Agent -- ingest)
    - Set FOMC_AzureOpenAI__ApiKey and FOMC_AzureOpenAI__Endpoint environment variables

Index types compared:
    FAISS HNSWFlat — graph-based ANN, same algorithm as Qdrant's internal index
    FAISS IVFFlat  — inverted file (cluster-based) ANN

Key tuning parameters:
    HNSW:  M (graph edges per node), efConstruction (build), efSearch (query)
    IVF:   nlist (# clusters), nprobe (clusters to visit at query time)
"""

from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import faiss
from qdrant_client import QdrantClient
from openai import AzureOpenAI

# ── Configuration ──────────────────────────────────────────────────────────────

QDRANT_HOST   = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT   = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION    = os.getenv("QDRANT_COLLECTION", "fomc_documents")

# Azure OpenAI - configure via environment variables
AZURE_ENDPOINT = os.getenv("FOMC_AzureOpenAI__Endpoint", "https://your-resource-name.openai.azure.com")
AZURE_API_KEY  = os.getenv("FOMC_AzureOpenAI__ApiKey", "your-api-key-here")
EMBED_DEPLOY   = os.getenv("FOMC_AzureOpenAI__EmbeddingDeployment", "text-embedding-3-large")
EMBED_DIMS     = int(os.getenv("FOMC_AzureOpenAI__EmbeddingDimensions", "3072"))

TOP_K          = 10
NUM_QUERIES    = 20
HNSW_M         = 32          # edges per node
IVF_NLIST_SQRT = True        # nlist = sqrt(N) — standard heuristic

# ── Query texts (analyst-style) ────────────────────────────────────────────────

QUERY_TEXTS = [
    "federal funds rate target range policy decision",
    "inflation outlook consumer price index PCE",
    "labor market employment unemployment rate",
    "quantitative tightening balance sheet reduction",
    "financial stability banking sector stress",
    "GDP growth economic activity projections",
    "monetary policy tightening restrictive stance",
    "committee vote unanimous dissent",
    "supply chain goods services prices",
    "wage growth nominal earnings workers",
    "housing market mortgage rate affordability",
    "global economic conditions international spillovers",
    "bank reserves monetary base liquidity",
    "payroll employment job creation labor demand",
    "core PCE price index inflation measure",
    "forward guidance future rate path projections",
    "neutral rate equilibrium long-run estimate",
    "uncertainty risk management decision framework",
    "credit conditions financial markets tightening",
    "soft landing recession risk scenario analysis",
][:NUM_QUERIES]

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    index_type: str
    params: dict[str, Any]
    build_time_s: float
    recalls: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def mean_recall(self) -> float:
        return float(np.mean(self.recalls)) if self.recalls else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0


# ── Step 1: Export vectors from Qdrant ────────────────────────────────────────

def export_vectors_from_qdrant() -> tuple[np.ndarray, list[str]]:
    """Scroll all vectors from the Qdrant collection."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}, collection='{COLLECTION}'")
    info = client.get_collection(COLLECTION)
    total = info.points_count
    print(f"  {total} points in collection, vector size={EMBED_DIMS}")

    vectors, ids = [], []
    offset = None

    while True:
        response = client.scroll(
            collection_name=COLLECTION,
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=True,
        )
        pts, next_offset = response

        if not pts:
            break

        for pt in pts:
            v = pt.vector
            if v is not None:
                # Handle named vectors (dict) vs single vector (list)
                if isinstance(v, dict):
                    # Use the 'dense' vector for semantic search comparison
                    dense_vec = v.get("dense")
                    if dense_vec is not None:
                        vectors.append(np.array(dense_vec, dtype=np.float32))
                        ids.append(str(pt.id))
                else:
                    vectors.append(np.array(v, dtype=np.float32))
                    ids.append(str(pt.id))

        if next_offset is None:
            break
        offset = next_offset

    mat = np.vstack(vectors).astype(np.float32)
    # Normalise for cosine similarity (faiss uses L2 on unit vectors)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.clip(norms, 1e-10, None)

    print(f"  Exported {mat.shape[0]} vectors  shape={mat.shape}")
    return mat, ids


# ── Step 2: Embed query texts ──────────────────────────────────────────────────

def embed_queries() -> np.ndarray:
    """Generate embeddings for the benchmark query texts via Azure AI Foundry."""
    if not AZURE_API_KEY:
        print("FOMC_AzureOpenAI__ApiKey not set — using random query vectors.")
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((len(QUERY_TEXTS), EMBED_DIMS)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version="2024-10-21",
    )
    print(f"Embedding {len(QUERY_TEXTS)} queries via {EMBED_DEPLOY}…")
    resp = client.embeddings.create(
        model=EMBED_DEPLOY,
        input=QUERY_TEXTS,
        dimensions=EMBED_DIMS,
    )
    vecs = np.array([e.embedding for e in resp.data], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


# ── Step 3: Build indexes ──────────────────────────────────────────────────────

def build_flat(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Brute-force exact index — ground truth."""
    idx = faiss.IndexFlatIP(vectors.shape[1])   # inner product == cosine for unit vecs
    idx.add(vectors)
    return idx


def build_hnsw(vectors: np.ndarray, M: int = HNSW_M) -> tuple[faiss.IndexHNSWFlat, float]:
    """FAISS HNSWFlat — graph-based ANN."""
    t0 = time.perf_counter()
    idx = faiss.IndexHNSWFlat(vectors.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = 200
    idx.add(vectors)
    build_s = time.perf_counter() - t0
    return idx, build_s


def build_ivf(vectors: np.ndarray, nlist: int | None = None) -> tuple[faiss.IndexIVFFlat, float]:
    """FAISS IVFFlat — cluster-based (Voronoi) ANN."""
    n = vectors.shape[0]
    nlist = nlist or max(1, int(n ** 0.5))   # sqrt heuristic
    quantizer = faiss.IndexFlatIP(vectors.shape[1])
    t0 = time.perf_counter()
    idx = faiss.IndexIVFFlat(quantizer, vectors.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    idx.train(vectors)
    idx.add(vectors)
    build_s = time.perf_counter() - t0
    return idx, build_s, nlist


# ── Step 4: Run queries and compute recall ─────────────────────────────────────

def compute_ground_truth(flat_idx: faiss.IndexFlatIP, queries: np.ndarray) -> np.ndarray:
    _, I = flat_idx.search(queries, TOP_K)
    return I   # shape (num_queries, TOP_K)


def benchmark_index(
    idx,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    result: BenchmarkResult,
) -> None:
    for i, q in enumerate(queries):
        q_vec = q.reshape(1, -1)
        t0 = time.perf_counter()
        _, I = idx.search(q_vec, TOP_K)
        latency_ms = (time.perf_counter() - t0) * 1000

        found    = set(I[0].tolist())
        expected = set(ground_truth[i].tolist())
        recall   = len(found & expected) / TOP_K

        result.recalls.append(recall)
        result.latencies_ms.append(latency_ms)


# ── Step 5: Print results ─────────────────────────────────────────────────────

def print_table(results: list[BenchmarkResult], n_vectors: int) -> None:
    print(f"\n{'─' * 78}")
    print(f"  Benchmark: {NUM_QUERIES} queries, top-{TOP_K}, corpus size = {n_vectors}")
    print(f"{'─' * 78}")
    print(f"  {'Index':<28} {'Recall@' + str(TOP_K):<14} {'Avg (ms)':<12} "
          f"{'P95 (ms)':<12} {'Build (s)':<10}")
    print(f"{'─' * 78}")
    for r in results:
        param_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
        print(f"  {r.index_type + ' (' + param_str + ')':<28} "
              f"{r.mean_recall:<14.3f} {r.mean_latency_ms:<12.2f} "
              f"{r.p95_latency_ms:<12.2f} {r.build_time_s:<10.2f}")
    print(f"{'─' * 78}")

    print("""
Key takeaways
─────────────
HNSW
  + Excellent recall with small ef; no training step required
  + Incrementally updatable (add points without full rebuild)
  + Memory: O(N × M × 4 bytes) — 16 edges → ~100 MB per 1M 1536-d vecs
  - Build time scales super-linearly with M and efConstruction
  - No support for filtered search without post-filtering overhead
  ✓ Best for: real-time RAG, low-latency retrieval, growing corpora

IVF (Inverted File)
  + Fast at high N — query touches only nprobe/nlist fraction of data
  + Lower memory per vector than HNSW
  + Supports efficient filtered search via inverted lists
  - Requires training (cluster assignment); rebuild on major distribution shift
  - Recall degrades if query falls near cluster boundaries (nprobe mitigates)
  ✓ Best for: very large static corpora (>10M), batch retrieval, analytics

Qdrant
  Uses HNSW internally with cosine similarity and optional scalar quantization.
  The C# benchmark (dotnet run -- benchmark) shows Qdrant ef-search recall sweep.
  ef=64 → typically >0.96 recall at <5 ms for this corpus size.
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Export corpus
    vectors, _ = export_vectors_from_qdrant()
    queries     = embed_queries()

    n, d = vectors.shape
    print(f"\nBuilding indexes on {n} vectors, dim={d}\n")

    # Ground truth
    flat_idx     = build_flat(vectors)
    ground_truth = compute_ground_truth(flat_idx, queries)

    results: list[BenchmarkResult] = []

    # ── HNSW at different efSearch values ─────────────────────────────────────
    hnsw_idx, hnsw_build = build_hnsw(vectors, M=HNSW_M)
    print(f"HNSW built in {hnsw_build:.2f}s (M={HNSW_M}, efConstruction=200)")

    for ef in [32, 64, 128, 256]:
        hnsw_idx.hnsw.efSearch = ef
        r = BenchmarkResult(
            index_type="FAISS HNSWFlat",
            params={"M": HNSW_M, "efSearch": ef},
            build_time_s=hnsw_build,
        )
        benchmark_index(hnsw_idx, queries, ground_truth, r)
        results.append(r)

    # ── IVF at different nprobe values ────────────────────────────────────────
    ivf_idx, ivf_build, nlist = build_ivf(vectors)
    print(f"IVF built in {ivf_build:.2f}s (nlist={nlist})")

    for nprobe in [1, 4, 16, 64]:
        if nprobe > nlist:
            continue
        ivf_idx.nprobe = nprobe
        r = BenchmarkResult(
            index_type="FAISS IVFFlat",
            params={"nlist": nlist, "nprobe": nprobe},
            build_time_s=ivf_build,
        )
        benchmark_index(ivf_idx, queries, ground_truth, r)
        results.append(r)

    # Print comparison table
    print_table(results, n)

    # Save results to JSON
    out = {
        "corpus_size": n,
        "vector_dims": d,
        "top_k": TOP_K,
        "num_queries": NUM_QUERIES,
        "results": [
            {
                "index_type":    r.index_type,
                "params":        r.params,
                "mean_recall":   round(r.mean_recall, 4),
                "mean_latency_ms": round(r.mean_latency_ms, 3),
                "p95_latency_ms":  round(r.p95_latency_ms, 3),
                "build_time_s":  round(r.build_time_s, 3),
            }
            for r in results
        ]
    }
    # Save to same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
