using Azure;
using Azure.AI.OpenAI;
using FOMC.Shared.Configuration;
using Microsoft.Extensions.Options;
using OpenAI.Chat;
using OpenAI.Embeddings;

namespace FOMC.Data.Factories;

/// <summary>
/// Factory for creating Azure OpenAI clients.
/// 
/// WHY THIS FACTORY:
/// - Centralizes credential management (DRY)
/// - Single point of configuration validation
/// - Easier to mock in tests
/// - Supports future scenarios like client pooling or caching
/// 
/// SDK: Azure.AI.OpenAI (GA - officially supports Azure AI Foundry)
/// Uses the unified AzureOpenAIClient which provides ChatClient and EmbeddingClient.
/// </summary>
public sealed class AzureOpenAIClientFactory : IAzureOpenAIClientFactory
{
    private readonly AzureOpenAIOptions _options;
    private readonly AzureOpenAIClient _client;

    public AzureOpenAIClientFactory(IOptions<AzureOpenAIOptions> options)
    {
        _options = options.Value;

        if (string.IsNullOrWhiteSpace(_options.Endpoint))
            throw new InvalidOperationException("AzureOpenAI:Endpoint is required.");

        if (string.IsNullOrWhiteSpace(_options.ApiKey))
            throw new InvalidOperationException("AzureOpenAI:ApiKey is required.");

        var endpoint = new Uri(_options.Endpoint);
        var credential = new AzureKeyCredential(_options.ApiKey);
        _client = new AzureOpenAIClient(endpoint, credential);
    }

    /// <inheritdoc/>
    public EmbeddingClient CreateEmbeddingsClient() =>
        _client.GetEmbeddingClient(_options.EmbeddingDeployment);

    /// <inheritdoc/>
    public ChatClient CreateChatClient() =>
        _client.GetChatClient(_options.ChatDeployment);

    /// <inheritdoc/>
    public string EmbeddingModel => _options.EmbeddingDeployment;

    /// <inheritdoc/>
    public string ChatModel => _options.ChatDeployment;

    /// <inheritdoc/>
    public int EmbeddingDimensions => _options.EmbeddingDimensions;

    /// <inheritdoc/>
    public int EmbeddingBatchSize => _options.EmbeddingBatchSize;
}

/// <summary>
/// Interface for Azure OpenAI client factory.
/// Enables testability and follows DIP.
/// </summary>
public interface IAzureOpenAIClientFactory
{
    EmbeddingClient CreateEmbeddingsClient();
    ChatClient CreateChatClient();
    string EmbeddingModel { get; }
    string ChatModel { get; }
    int EmbeddingDimensions { get; }
    int EmbeddingBatchSize { get; }
}
