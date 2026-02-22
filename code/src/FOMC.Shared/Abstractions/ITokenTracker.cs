using FOMC.Shared.Models;

namespace FOMC.Shared.Abstractions;

/// <summary>
/// Abstraction for tracking API token usage across services.
/// Single Responsibility: token counting (not cost calculation or reporting).
/// </summary>
public interface ITokenTracker
{
    /// <summary>
    /// Record embedding tokens consumed.
    /// </summary>
    void AddEmbeddingTokens(int tokens);

    /// <summary>
    /// Record chat/completion tokens consumed.
    /// </summary>
    void AddChatTokens(int promptTokens, int completionTokens);

    /// <summary>
    /// Get current usage snapshot (thread-safe).
    /// </summary>
    TokenUsageSnapshot Snapshot();

    /// <summary>
    /// Reset all counters.
    /// </summary>
    void Reset();
}
