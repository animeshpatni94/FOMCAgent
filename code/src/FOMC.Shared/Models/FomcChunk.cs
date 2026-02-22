namespace FOMC.Shared.Models;

/// <summary>
/// Represents a single chunk of an FOMC document with full metadata for grounding and citation.
/// </summary>
public record FomcChunk
{
    /// <summary>Unique identifier (UUID), deterministic from doc+chunk index.</summary>
    public required string ChunkId { get; init; }

    /// <summary>"press_statement" or "minutes"</summary>
    public required string DocType { get; init; }

    /// <summary>ISO date of the meeting, e.g. "2024-01-31"</summary>
    public required string MeetingDate { get; init; }

    /// <summary>Original Fed Reserve URL.</summary>
    public required string SourceUrl { get; init; }

    /// <summary>The actual text content of this chunk.</summary>
    public required string Text { get; init; }

    /// <summary>Zero-based position within the parent document.</summary>
    public required int ChunkIndex { get; init; }

    /// <summary>Total number of chunks in the parent document.</summary>
    public required int TotalChunks { get; init; }

    /// <summary>Section heading if detectable (e.g. "Economic Activity", "Inflation").</summary>
    public string? SectionTitle { get; init; }

    // ── Creative Enhancements ────────────────────────────────────────────────

    /// <summary>
    /// LLM-generated summary of this chunk (computed at ingestion).
    /// Embedding the summary alongside raw text improves semantic recall
    /// for high-level queries like "What was the Fed's stance on inflation?"
    /// </summary>
    public string? Summary { get; init; }

    /// <summary>
    /// Extracted key entities: rates, percentages, policy decisions, dates.
    /// Enables precise metadata filtering and improves answer grounding.
    /// </summary>
    public List<string>? KeyEntities { get; init; }

    /// <summary>
    /// High-level document summary (shared across all chunks in the doc).
    /// Provides context without needing to retrieve multiple chunks.
    /// </summary>
    public string? DocumentSummary { get; init; }

    /// <summary>
    /// Flag indicating this chunk contains key policy decisions.
    /// The agent can prioritize these chunks for policy-focused queries.
    /// </summary>
    public bool IsKeyPassage { get; init; }

    /// <summary>Human-readable document label for citations.</summary>
    public string DocumentLabel => DocType switch
    {
        "press_statement" => $"FOMC Press Statement ({MeetingDate})",
        "minutes"         => $"FOMC Meeting Minutes ({MeetingDate})",
        _                 => $"FOMC Document ({MeetingDate})"
    };
}
