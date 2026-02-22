using System.Text.Json;

namespace FOMC.Shared.Utilities;

/// <summary>
/// Shared JSON serialization settings for consistent output.
/// </summary>
public static class JsonHelpers
{
    /// <summary>
    /// Compact JSON options (no indentation) for MCP tool responses.
    /// </summary>
    public static readonly JsonSerializerOptions CompactOptions = new()
    {
        WriteIndented = false,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    /// <summary>
    /// Pretty-printed JSON for debugging/logging.
    /// </summary>
    public static readonly JsonSerializerOptions PrettyOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    /// <summary>
    /// Serialize to compact JSON string.
    /// </summary>
    public static string ToJson<T>(T value) =>
        JsonSerializer.Serialize(value, CompactOptions);

    /// <summary>
    /// Serialize to pretty-printed JSON string.
    /// </summary>
    public static string ToPrettyJson<T>(T value) =>
        JsonSerializer.Serialize(value, PrettyOptions);
}
