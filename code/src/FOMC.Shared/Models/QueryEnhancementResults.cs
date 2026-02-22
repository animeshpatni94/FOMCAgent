namespace FOMC.Shared.Models;

/// <summary>
/// Result of HyDE (Hypothetical Document Embedding) query enhancement.
/// </summary>
public record HydeResult
{
    public required string OriginalQuery { get; init; }
    public string? HypotheticalDocument { get; init; }
    public required float[] Embedding { get; init; }
    public required string Strategy { get; init; }
}

/// <summary>
/// Result of query decomposition into sub-questions.
/// </summary>
public record QueryDecomposition
{
    public required string OriginalQuery { get; init; }
    public required List<string> SubQueries { get; init; }
    public required string Strategy { get; init; }
}

/// <summary>
/// Result of query expansion into multiple phrasings.
/// </summary>
public record QueryExpansion
{
    public required string OriginalQuery { get; init; }
    public required List<string> ExpandedQueries { get; init; }
    public required string Strategy { get; init; }
}

/// <summary>
/// Result of step-back query generation.
/// </summary>
public record StepBackResult
{
    public required string OriginalQuery { get; init; }
    public string? StepBackQuery { get; init; }
    public required string Strategy { get; init; }
}
