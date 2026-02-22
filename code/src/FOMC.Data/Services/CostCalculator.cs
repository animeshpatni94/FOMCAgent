using FOMC.Shared.Abstractions;
using FOMC.Shared.Configuration;
using FOMC.Shared.Models;
using Microsoft.Extensions.Options;

namespace FOMC.Data.Services;

/// <summary>
/// Calculates API costs from token usage.
/// Separates cost calculation concern from token tracking (SRP).
/// </summary>
public sealed class CostCalculator : ICostCalculator
{
    private readonly PricingOptions _pricing;

    public CostCalculator(IOptions<PricingOptions> pricing)
    {
        _pricing = pricing.Value;
    }

    /// <inheritdoc/>
    public decimal CalculateCostUsd(TokenUsageSnapshot usage) =>
        (usage.EmbeddingTokens / 1000m * _pricing.EmbeddingPer1KTokens) +
        (usage.ChatPromptTokens / 1000m * _pricing.ChatInputPer1KTokens) +
        (usage.ChatCompletionTokens / 1000m * _pricing.ChatOutputPer1KTokens);

    /// <inheritdoc/>
    public string FormatCostReport(TokenUsageSnapshot usage)
    {
        var cost = CalculateCostUsd(usage);
        
        return $"""

        ── Token Usage ──────────────────────────────
          Embedding (text-embedding-3-large): {usage.EmbeddingTokens:N0}
          Chat prompt:                        {usage.ChatPromptTokens:N0}
          Chat completion:                    {usage.ChatCompletionTokens:N0}
          Total tokens:                       {usage.TotalTokens:N0}
          API calls:                          {usage.ApiCallCount:N0}
          Estimated cost:                     ${cost:F4}
        ─────────────────────────────────────────────
        """;
    }
}
