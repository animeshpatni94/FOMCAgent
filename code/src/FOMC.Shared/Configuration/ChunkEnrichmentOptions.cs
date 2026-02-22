namespace FOMC.Shared.Configuration;

/// <summary>
/// Configuration for chunk enrichment during ingestion.
/// </summary>
public sealed class ChunkEnrichmentOptions
{
    public const string SectionName = "ChunkEnrichment";

    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Temperature for summary generation.
    /// </summary>
    public float SummaryTemperature { get; set; } = 0.3f;

    /// <summary>
    /// Max tokens for chunk summary.
    /// </summary>
    public int SummaryMaxTokens { get; set; } = 100;

    /// <summary>
    /// Max tokens for document-level summary.
    /// </summary>
    public int DocumentSummaryMaxTokens { get; set; } = 200;
}
