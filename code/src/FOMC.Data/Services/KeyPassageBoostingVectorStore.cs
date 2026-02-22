using FOMC.Shared.Abstractions;
using FOMC.Shared.Models;

namespace FOMC.Data.Services;

/// <summary>
/// Decorator that applies a score boost to key passages (policy decisions, votes).
/// 
/// This implements the Decorator Pattern to add boosting behavior without modifying
/// the underlying vector store implementation (Open/Closed Principle).
/// 
/// Why boosting instead of filtering?
/// - Hard filtering (IsKeyPassage == true) can introduce bias by excluding
///   relevant context like economic outlook, housing discussion, etc.
/// - Soft boosting prioritizes key passages when scores are close, but allows
///   highly relevant non-policy content to surface if semantic match is strong.
/// </summary>
public class KeyPassageBoostingVectorStore : IVectorStore
{
    private readonly IVectorStore _inner;
    
    /// <summary>
    /// Boost applied to key passages during ranking.
    /// Small value (0.05) ensures relevance still dominates — key passages only
    /// surface higher when semantic scores are already close.
    /// </summary>
    private const float KeyPassageBoost = 0.05f;

    public KeyPassageBoostingVectorStore(IVectorStore inner)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
    }

    /// <summary>
    /// Search with key passage boosting applied.
    /// Fetches extra candidates, applies boost, re-ranks, and returns top-k.
    /// </summary>
    public async Task<IReadOnlyList<SearchResult>> SearchAsync(
        float[] queryVector,
        int topK = 5,
        string? docType = null,
        string? dateFrom = null,
        string? dateTo = null,
        int hnswEf = 64,
        bool exact = false,
        CancellationToken ct = default)
    {
        // Fetch extra candidates to allow re-ranking to surface boosted results
        var candidateCount = Math.Min(topK + 10, topK * 2);
        
        var rawResults = await _inner.SearchAsync(
            queryVector, candidateCount, docType, dateFrom, dateTo, hnswEf, exact, ct);

        if (rawResults.Count == 0)
            return rawResults;

        // Apply boost and re-rank
        var boosted = rawResults
            .Select(r => r with { Score = r.Score + (r.Chunk.IsKeyPassage ? KeyPassageBoost : 0f) })
            .OrderByDescending(r => r.Score)
            .Take(topK)
            .ToList();

        return boosted;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Pass-through methods (Decorator pattern - delegate to inner)
    // ══════════════════════════════════════════════════════════════════════════

    public Task EnsureCollectionAsync(int vectorSize, CancellationToken ct = default)
        => _inner.EnsureCollectionAsync(vectorSize, ct);

    public Task DeleteCollectionAsync(CancellationToken ct = default)
        => _inner.DeleteCollectionAsync(ct);

    public Task UpsertAsync(IEnumerable<(FomcChunk Chunk, float[] Vector)> items, CancellationToken ct = default)
        => _inner.UpsertAsync(items, ct);

    /// <summary>
    /// Hybrid search with key passage boosting applied.
    /// Combines dense vectors + keyword search, then applies boost to key passages.
    /// </summary>
    public async Task<IReadOnlyList<SearchResult>> HybridSearchAsync(
        float[] queryVector,
        string queryText,
        int topK = 5,
        float denseWeight = 0.7f,
        string? docType = null,
        string? dateFrom = null,
        string? dateTo = null,
        CancellationToken ct = default)
    {
        // Fetch extra candidates for re-ranking
        var candidateCount = Math.Min(topK + 10, topK * 2);
        
        var rawResults = await _inner.HybridSearchAsync(
            queryVector, queryText, candidateCount, denseWeight, docType, dateFrom, dateTo, ct);

        if (rawResults.Count == 0)
            return rawResults;

        // Apply key passage boost and re-rank
        var boosted = rawResults
            .Select(r => r with { Score = r.Score + (r.Chunk.IsKeyPassage ? KeyPassageBoost : 0f) })
            .OrderByDescending(r => r.Score)
            .Take(topK)
            .ToList();

        return boosted;
    }

    public Task<long> CountAsync(CancellationToken ct = default)
        => _inner.CountAsync(ct);

    public Task<List<string>> GetMeetingDatesAsync(CancellationToken ct = default)
        => _inner.GetMeetingDatesAsync(ct);

    public Task<FomcChunk?> GetChunkAsync(string chunkId, CancellationToken ct = default)
        => _inner.GetChunkAsync(chunkId, ct);
}
