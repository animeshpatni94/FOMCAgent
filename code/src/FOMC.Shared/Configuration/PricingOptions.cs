namespace FOMC.Shared.Configuration;

/// <summary>
/// Pricing configuration for cost estimation.
/// Allows updating prices without code changes.
/// </summary>
public sealed class PricingOptions
{
    public const string SectionName = "Pricing";

    /// <summary>
    /// Cost per 1K embedding tokens (text-embedding-3-large).
    /// </summary>
    public decimal EmbeddingPer1KTokens { get; set; } = 0.00013m;

    /// <summary>
    /// Cost per 1K chat input tokens.
    /// </summary>
    public decimal ChatInputPer1KTokens { get; set; } = 0.01m;

    /// <summary>
    /// Cost per 1K chat output tokens.
    /// </summary>
    public decimal ChatOutputPer1KTokens { get; set; } = 0.03m;
}
