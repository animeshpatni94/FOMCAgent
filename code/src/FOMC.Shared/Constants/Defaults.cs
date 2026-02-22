namespace FOMC.Shared.Constants;

/// <summary>
/// Default values used across the application.
/// Centralizes magic numbers for consistency and easy tuning.
/// </summary>
public static class Defaults
{
    // ── Search defaults ────────────────────────────────────────────────────
    public const int DefaultTopK = 5;
    public const int MaxTopK = 10;
    public const int MinTopK = 1;

    // ── HNSW defaults ──────────────────────────────────────────────────────
    public const int DefaultHnswM = 16;
    public const int DefaultHnswEfConstruct = 100;
    public const int DefaultSearchEf = 64;

    // ── Embedding defaults ─────────────────────────────────────────────────
    public const int DefaultEmbeddingDimensions = 3072;
    public const int EmbeddingBatchSize = 64;

    // ── Chunking defaults ──────────────────────────────────────────────────
    public const int DefaultChunkSize = 800;
    public const int DefaultChunkOverlap = 100;

    // ══════════════════════════════════════════════════════════════════════
    // LLM OUTPUT TOKEN LIMITS
    // 
    // Engineering rationale for each limit:
    // - Smaller limits = faster response, lower cost, less risk of rambling
    // - Each task has a specific output size requirement
    // ══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Agent final response - needs enough room for detailed analysis with citations.
    /// </summary>
    public const int AgentResponseMaxTokens = 2048;

    /// <summary>
    /// CRAG retrieval evaluation - structured JSON output with explanations.
    /// </summary>
    public const int RetrievalEvaluationMaxTokens = 800;

    /// <summary>
    /// Chunk enrichment (summaries) - concise 1-2 sentence summaries.
    /// </summary>
    public const int ChunkSummaryMaxTokens = 150;

    /// <summary>
    /// Document-level summary - brief executive summary.
    /// </summary>
    public const int DocumentSummaryMaxTokens = 300;

    /// <summary>
    /// HyDE hypothetical document - short paragraph mimicking FOMC style.
    /// </summary>
    public const int HydeMaxTokens = 200;

    /// <summary>
    /// Query decomposition - list of 2-4 sub-questions.
    /// </summary>
    public const int QueryDecompositionMaxTokens = 200;

    /// <summary>
    /// Query expansion - list of 2-3 alternative phrasings.
    /// </summary>
    public const int QueryExpansionMaxTokens = 150;

    /// <summary>
    /// Step-back query - single broader question.
    /// </summary>
    public const int StepBackMaxTokens = 100;

    /// <summary>
    /// Query refinement (CRAG) - single refined query.
    /// </summary>
    public const int QueryRefinementMaxTokens = 100;

    /// <summary>
    /// Reranking scores - structured JSON with scores.
    /// </summary>
    public const int RerankMaxTokens = 200;

    /// <summary>
    /// Evaluation scoring - brief relevance assessment.
    /// </summary>
    public const int EvaluationMaxTokens = 256;

    // ── Temperature settings ───────────────────────────────────────────────
    
    /// <summary>
    /// Default temperature for general tasks.
    /// </summary>
    public const float DefaultTemperature = 0.7f;

    /// <summary>
    /// Low temperature for deterministic tasks (evaluation, reranking).
    /// </summary>
    public const float DeterministicTemperature = 0.0f;

    /// <summary>
    /// Medium temperature for creative but grounded tasks (HyDE).
    /// </summary>
    public const float CreativeTemperature = 0.7f;

    /// <summary>
    /// Very low temperature for factual agent responses.
    /// </summary>
    public const float AgentTemperature = 0.1f;

    // ── Confidence thresholds ──────────────────────────────────────────────
    public const float HighConfidenceThreshold = 0.85f;
    public const float MediumConfidenceThreshold = 0.75f;
    public const float LowConfidenceThreshold = 0.65f;

    // ── Security limits ────────────────────────────────────────────────────
    
    /// <summary>
    /// Maximum query length to prevent context overflow attacks.
    /// </summary>
    public const int MaxQueryLength = 2000;

    /// <summary>
    /// Maximum iterations for agent ReAct loop.
    /// </summary>
    public const int MaxAgentIterations = 6;
}
