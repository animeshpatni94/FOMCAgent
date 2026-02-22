namespace FOMC.Shared.Models;

/// <summary>
/// Result of one RAG evaluation run against a single question.
/// </summary>
public record EvaluationResult
{
    public required string Question         { get; init; }
    public required string ExpectedAnswer   { get; init; }
    public required string ActualAnswer     { get; init; }

    /// <summary>0–1: fraction of answer claims supported by retrieved passages.</summary>
    public float FaithfulnessScore          { get; init; }

    /// <summary>0–1: fraction of retrieved chunks that are relevant to the question.</summary>
    public float ContextRelevanceScore      { get; init; }

    /// <summary>0–1: lexical/semantic similarity to expected answer.</summary>
    public float AnswerSimilarityScore      { get; init; }

    public int   ToolCallsUsed              { get; init; }
    public int   SourcesRetrieved           { get; init; }
    public string? JudgeReasoning           { get; init; }

    public float OverallScore =>
        (FaithfulnessScore + ContextRelevanceScore + AnswerSimilarityScore) / 3f;
}
