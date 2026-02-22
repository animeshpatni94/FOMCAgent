using FOMC.Shared.Models;

namespace FOMC.Shared.Abstractions;

/// <summary>
/// Abstraction for calculating API costs from token usage.
/// Separates cost calculation concern from token tracking (SRP).
/// </summary>
public interface ICostCalculator
{
    /// <summary>
    /// Calculate estimated cost in USD from a token usage snapshot.
    /// </summary>
    decimal CalculateCostUsd(TokenUsageSnapshot usage);

    /// <summary>
    /// Generate a formatted cost report.
    /// </summary>
    string FormatCostReport(TokenUsageSnapshot usage);
}
