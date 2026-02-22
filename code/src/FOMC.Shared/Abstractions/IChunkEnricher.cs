using FOMC.Shared.Models;

namespace FOMC.Shared.Abstractions;

/// <summary>
/// Abstraction for chunk enrichment during document ingestion.
/// </summary>
public interface IChunkEnricher
{
    /// <summary>
    /// Enrich chunks with LLM-generated summaries, entities, and key passage flags.
    /// </summary>
    Task<List<FomcChunk>> EnrichAsync(
        List<FomcChunk> chunks,
        string? documentSummary = null,
        CancellationToken ct = default);

    /// <summary>
    /// Generate a summary for an entire document.
    /// </summary>
    Task<string?> GenerateDocumentSummaryAsync(
        List<FomcChunk> chunks,
        CancellationToken ct = default);
}
