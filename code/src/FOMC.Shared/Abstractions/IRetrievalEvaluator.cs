namespace FOMC.Shared.Abstractions;

/// <summary>
/// CRAG-style retrieval evaluator: assesses whether retrieved documents
/// are relevant enough to answer the query, or if fallback strategies are needed.
///
/// This implements the Corrective RAG (CRAG) pattern:
/// - CORRECT: Results are highly relevant, proceed with generation
/// - AMBIGUOUS: Results are partially relevant, may need refinement
/// - INCORRECT: Results are not relevant, trigger fallback (web search, query rewrite)
///
/// Reference: "Corrective Retrieval Augmented Generation" (Yan et al., 2024)
/// </summary>
public interface IRetrievalEvaluator
{
    /// <summary>
    /// Evaluate the relevance of retrieved results to the query.
    /// </summary>
    Task<RetrievalEvaluation> EvaluateAsync(
        string query,
        IReadOnlyList<EvaluationCandidate> candidates,
        CancellationToken ct = default);

    /// <summary>
    /// Get a refined query if initial retrieval was ambiguous or incorrect.
    /// </summary>
    Task<string?> GetRefinedQueryAsync(
        string originalQuery,
        RetrievalEvaluation evaluation,
        CancellationToken ct = default);
}

/// <summary>
/// Input candidate for evaluation.
/// </summary>
public record EvaluationCandidate(
    string ChunkId,
    string Text,
    string? Summary,
    float Score,
    bool IsKeyPassage);

/// <summary>
/// Result of retrieval evaluation.
/// </summary>
public record RetrievalEvaluation
{
    /// <summary>
    /// Overall verdict: Correct, Ambiguous, or Incorrect.
    /// </summary>
    public required RetrievalVerdict Verdict { get; init; }

    /// <summary>
    /// Confidence score (0-1) in the verdict.
    /// </summary>
    public required float Confidence { get; init; }

    /// <summary>
    /// Human-readable explanation of the evaluation.
    /// </summary>
    public required string Explanation { get; init; }

    /// <summary>
    /// Per-candidate relevance scores (0-5).
    /// </summary>
    public required IReadOnlyList<CandidateRelevance> CandidateScores { get; init; }

    /// <summary>
    /// Suggested action if results are inadequate.
    /// </summary>
    public string? SuggestedAction { get; init; }
}

/// <summary>
/// Per-candidate relevance assessment.
/// </summary>
public record CandidateRelevance(
    string ChunkId,
    int RelevanceScore,
    string Rationale);

/// <summary>
/// CRAG verdict categories.
/// </summary>
public enum RetrievalVerdict
{
    /// <summary>
    /// Results are highly relevant to the query. Proceed with answer generation.
    /// </summary>
    Correct,

    /// <summary>
    /// Results are partially relevant. May need query refinement or additional search.
    /// </summary>
    Ambiguous,

    /// <summary>
    /// Results are not relevant. Trigger fallback strategy.
    /// </summary>
    Incorrect
}
