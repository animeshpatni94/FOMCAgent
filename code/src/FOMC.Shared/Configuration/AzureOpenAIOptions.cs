using System.ComponentModel.DataAnnotations;

namespace FOMC.Shared.Configuration;

/// <summary>
/// Strongly-typed configuration for Azure OpenAI services.
/// Bind from appsettings.json section "AzureOpenAI".
/// </summary>
public sealed class AzureOpenAIOptions
{
    public const string SectionName = "AzureOpenAI";

    [Required]
    public string Endpoint { get; set; } = string.Empty;

    [Required]
    public string ApiKey { get; set; } = string.Empty;

    public string EmbeddingDeployment { get; set; } = "text-embedding-3-large";

    public int EmbeddingDimensions { get; set; } = 3072;

    public string ChatDeployment { get; set; } = "gpt-5-chat";

    public int EmbeddingBatchSize { get; set; } = 64;
}
