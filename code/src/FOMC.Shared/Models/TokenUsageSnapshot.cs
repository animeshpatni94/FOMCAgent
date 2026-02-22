namespace FOMC.Shared.Models;

/// <summary>
/// Thread-safe snapshot of token usage across all API calls.
/// </summary>
public sealed record TokenUsageSnapshot(
    long EmbeddingTokens,
    long ChatPromptTokens,
    long ChatCompletionTokens,
    long ApiCallCount)
{
    public long TotalChatTokens => ChatPromptTokens + ChatCompletionTokens;
    public long TotalTokens => EmbeddingTokens + TotalChatTokens;
}
