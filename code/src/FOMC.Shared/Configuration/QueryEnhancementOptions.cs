namespace FOMC.Shared.Configuration;

/// <summary>
/// Configuration for query enhancement features.
/// </summary>
public sealed class QueryEnhancementOptions
{
    public const string SectionName = "QueryEnhancement";

    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Temperature for HyDE hypothetical document generation.
    /// </summary>
    public float HydeTemperature { get; set; } = 0.7f;

    /// <summary>
    /// Temperature for query decomposition.
    /// </summary>
    public float DecompositionTemperature { get; set; } = 0.3f;

    /// <summary>
    /// Temperature for query expansion.
    /// </summary>
    public float ExpansionTemperature { get; set; } = 0.5f;

    /// <summary>
    /// Maximum tokens for HyDE hypothetical document.
    /// </summary>
    public int HydeMaxTokens { get; set; } = 200;

    /// <summary>
    /// K parameter for Reciprocal Rank Fusion.
    /// </summary>
    public int RrfK { get; set; } = 60;
}
