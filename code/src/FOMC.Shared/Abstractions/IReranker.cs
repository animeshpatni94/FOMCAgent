using FOMC.Shared.Models;

namespace FOMC.Shared.Abstractions;

/// <summary>
/// Abstraction for result reranking (cross-encoder or LLM-based).
/// </summary>
public interface IReranker
{
    /// <summary>
    /// Rerank search results for improved precision.
    /// </summary>
    Task<List<SearchResult>> RerankAsync(
        string query,
        IReadOnlyList<SearchResult> candidates,
        int topN = 5,
        CancellationToken ct = default);
}
