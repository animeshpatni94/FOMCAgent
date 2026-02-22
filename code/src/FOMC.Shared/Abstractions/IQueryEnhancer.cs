using FOMC.Shared.Models;

namespace FOMC.Shared.Abstractions;

/// <summary>
/// Abstraction for query enhancement techniques (HyDE, expansion, decomposition, step-back).
/// </summary>
public interface IQueryEnhancer
{
    /// <summary>
    /// Generate a hypothetical document embedding (HyDE technique).
    /// </summary>
    Task<HydeResult> GenerateHydeEmbeddingAsync(string query, CancellationToken ct = default);

    /// <summary>
    /// Decompose a complex query into sub-questions.
    /// </summary>
    Task<QueryDecomposition> DecomposeQueryAsync(string query, CancellationToken ct = default);

    /// <summary>
    /// Expand query into multiple phrasings/synonyms.
    /// </summary>
    Task<QueryExpansion> ExpandQueryAsync(string query, CancellationToken ct = default);

    /// <summary>
    /// Generate a broader "step-back" query for context.
    /// </summary>
    Task<StepBackResult> GenerateStepBackQueryAsync(string query, CancellationToken ct = default);
}
