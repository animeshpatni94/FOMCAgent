namespace FOMC.Shared.Configuration;

/// <summary>
/// Configuration for reranking features.
/// </summary>
public sealed class RerankingOptions
{
    public const string SectionName = "Reranking";

    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Temperature for LLM-based reranking (should be 0 for deterministic scoring).
    /// </summary>
    public float Temperature { get; set; } = 0.0f;

    /// <summary>
    /// Maximum tokens for reranking response.
    /// </summary>
    public int MaxTokens { get; set; } = 200;
}
