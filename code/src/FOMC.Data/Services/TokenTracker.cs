using FOMC.Shared.Abstractions;
using FOMC.Shared.Models;

namespace FOMC.Data.Services;

/// <summary>
/// Thread-safe singleton that accumulates token usage across all API calls.
/// Implements ITokenTracker for testability.
///
/// Single Responsibility: Track token counts only.
/// Cost calculation is delegated to ICostCalculator (SRP).
/// </summary>
public sealed class TokenTracker : ITokenTracker
{
    private long _embeddingTokensTotal;
    private long _chatPromptTokensTotal;
    private long _chatCompletionTokensTotal;
    private long _apiCallCount;

    // ── Recording ─────────────────────────────────────────────────────────────

    public void AddEmbeddingTokens(int tokens)
    {
        Interlocked.Add(ref _embeddingTokensTotal, tokens);
        Interlocked.Increment(ref _apiCallCount);
    }

    public void AddChatTokens(int promptTokens, int completionTokens)
    {
        Interlocked.Add(ref _chatPromptTokensTotal, promptTokens);
        Interlocked.Add(ref _chatCompletionTokensTotal, completionTokens);
        Interlocked.Increment(ref _apiCallCount);
    }

    // ── Reading ───────────────────────────────────────────────────────────────

    public long EmbeddingTokens => Interlocked.Read(ref _embeddingTokensTotal);
    public long ChatPromptTokens => Interlocked.Read(ref _chatPromptTokensTotal);
    public long ChatCompletionTokens => Interlocked.Read(ref _chatCompletionTokensTotal);
    public long TotalChatTokens => ChatPromptTokens + ChatCompletionTokens;
    public long TotalTokens => EmbeddingTokens + TotalChatTokens;
    public long ApiCallCount => Interlocked.Read(ref _apiCallCount);

    public void Reset()
    {
        Interlocked.Exchange(ref _embeddingTokensTotal, 0);
        Interlocked.Exchange(ref _chatPromptTokensTotal, 0);
        Interlocked.Exchange(ref _chatCompletionTokensTotal, 0);
        Interlocked.Exchange(ref _apiCallCount, 0);
    }

    public TokenUsageSnapshot Snapshot() => new(
        EmbeddingTokens, ChatPromptTokens, ChatCompletionTokens, ApiCallCount);
}
