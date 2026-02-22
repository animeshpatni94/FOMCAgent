namespace FOMC.Shared.Abstractions;

/// <summary>
/// Abstraction for text embedding generation.
/// Enables testability and allows swapping embedding providers.
/// </summary>
public interface IEmbeddingService
{
    /// <summary>
    /// The dimensionality of the embedding vectors.
    /// </summary>
    int Dimensions { get; }

    /// <summary>
    /// Generate embedding for a single text (typically query-time).
    /// </summary>
    Task<float[]> GetEmbeddingAsync(string text, CancellationToken ct = default);

    /// <summary>
    /// Generate embeddings for multiple texts (batch ingestion).
    /// </summary>
    Task<float[][]> GetEmbeddingsAsync(IList<string> texts, CancellationToken ct = default);
}
