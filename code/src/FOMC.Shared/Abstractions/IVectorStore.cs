using FOMC.Shared.Models;

namespace FOMC.Shared.Abstractions;

/// <summary>
/// Abstraction for vector database operations.
/// Allows swapping Qdrant for another vector DB (Pinecone, Weaviate, etc.).
/// </summary>
public interface IVectorStore
{
    /// <summary>
    /// Ensure the vector collection exists with proper configuration.
    /// </summary>
    Task EnsureCollectionAsync(int vectorSize, CancellationToken ct = default);

    /// <summary>
    /// Delete the entire collection (used for re-ingestion).
    /// </summary>
    Task DeleteCollectionAsync(CancellationToken ct = default);

    /// <summary>
    /// Upsert chunks with their embedding vectors.
    /// </summary>
    Task UpsertAsync(IEnumerable<(FomcChunk Chunk, float[] Vector)> items, CancellationToken ct = default);

    /// <summary>
    /// Semantic vector search with optional filters.
    /// </summary>
    Task<IReadOnlyList<SearchResult>> SearchAsync(
        float[] queryVector,
        int topK = 5,
        string? docType = null,
        string? dateFrom = null,
        string? dateTo = null,
        int hnswEf = 64,
        bool exact = false,
        CancellationToken ct = default);

    /// <summary>
    /// Hybrid search combining dense vectors (semantic) with keyword matching (BM25-style).
    /// Uses Reciprocal Rank Fusion to merge results from both retrieval methods.
    /// Based on Anthropic's Contextual Retrieval research showing 49% improvement.
    /// </summary>
    /// <param name="queryVector">Dense embedding vector for semantic search</param>
    /// <param name="queryText">Original query text for keyword matching</param>
    /// <param name="topK">Number of results to return</param>
    /// <param name="denseWeight">Weight for dense results (0-1). Keyword weight = 1 - denseWeight</param>
    /// <param name="docType">Optional document type filter</param>
    /// <param name="dateFrom">Optional start date filter</param>
    /// <param name="dateTo">Optional end date filter</param>
    /// <param name="ct">Cancellation token</param>
    Task<IReadOnlyList<SearchResult>> HybridSearchAsync(
        float[] queryVector,
        string queryText,
        int topK = 5,
        float denseWeight = 0.7f,
        string? docType = null,
        string? dateFrom = null,
        string? dateTo = null,
        CancellationToken ct = default);

    /// <summary>
    /// Get total count of indexed documents.
    /// </summary>
    Task<long> CountAsync(CancellationToken ct = default);

    /// <summary>
    /// List all unique meeting dates in the corpus.
    /// </summary>
    Task<List<string>> GetMeetingDatesAsync(CancellationToken ct = default);

    /// <summary>
    /// Retrieve a specific chunk by its ID.
    /// </summary>
    Task<FomcChunk?> GetChunkAsync(string chunkId, CancellationToken ct = default);
}
