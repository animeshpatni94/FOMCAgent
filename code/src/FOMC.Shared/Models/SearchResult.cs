namespace FOMC.Shared.Models;

/// <summary>
/// A retrieved chunk with its cosine similarity score.
/// The Citation property is the canonical in-text reference format.
/// </summary>
public record SearchResult
{
    public required FomcChunk Chunk { get; init; }

    /// <summary>Cosine similarity score [0, 1]. Higher = more relevant.</summary>
    public required float Score { get; init; }

    /// <summary>
    /// Formatted citation string for inline use in agent responses.
    /// Example: [Source: FOMC Press Statement (2024-01-31), Chunk 2/5]
    /// </summary>
    public string Citation =>
        $"[Source: {Chunk.DocumentLabel}, Chunk {Chunk.ChunkIndex + 1}/{Chunk.TotalChunks}]";
}
