using FOMC.Data.Factories;
using FOMC.Shared.Abstractions;
using Microsoft.Extensions.Logging;
using OpenAI.Embeddings;

namespace FOMC.Data.Services;

/// <summary>
/// Wraps the Azure OpenAI EmbeddingClient.
/// Implements IEmbeddingService for testability and DIP compliance.
/// </summary>
public sealed class EmbeddingService : IEmbeddingService
{
    private readonly EmbeddingClient _client;
    private readonly string _modelName;
    private readonly int _dimensions;
    private readonly int _batchSize;
    private readonly ITokenTracker _tokens;
    private readonly ILogger<EmbeddingService> _logger;

    public int Dimensions => _dimensions;

    public EmbeddingService(
        IAzureOpenAIClientFactory clientFactory,
        ITokenTracker tokens,
        ILogger<EmbeddingService> logger)
    {
        _logger = logger;
        _tokens = tokens;
        _modelName = clientFactory.EmbeddingModel;
        _dimensions = clientFactory.EmbeddingDimensions;
        _batchSize = clientFactory.EmbeddingBatchSize;
        _client = clientFactory.CreateEmbeddingsClient();

        _logger.LogInformation(
            "EmbeddingService ready — model={Model} dims={Dims} batch={Batch}",
            _modelName, _dimensions, _batchSize);
    }

    /// <summary>Single embedding (query time).</summary>
    public async Task<float[]> GetEmbeddingAsync(string text, CancellationToken ct = default)
    {
        _logger.LogDebug("Generating single embedding for text ({Len} chars)...", text.Length);
        try
        {
            var options = new EmbeddingGenerationOptions { Dimensions = _dimensions };
            // Use batch API even for single text to get usage info
            var result = await _client.GenerateEmbeddingsAsync([text.Trim()], options, ct);
            _tokens.AddEmbeddingTokens(result.Value.Usage.InputTokenCount);
            _logger.LogDebug("Embedding generated successfully ({Tokens} tokens)", result.Value.Usage.InputTokenCount);
            return result.Value[0].ToFloats().ToArray();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to generate embedding: {Message}", ex.Message);
            throw;
        }
    }

    /// <summary>
    /// Batched embeddings (ingestion). Batches to stay within Azure limits.
    /// </summary>
    public async Task<float[][]> GetEmbeddingsAsync(
        IList<string> texts, CancellationToken ct = default)
    {
        _logger.LogDebug("Generating embeddings for {Count} texts in batches of {BatchSize}...", 
            texts.Count, _batchSize);
        var all = new List<float[]>(texts.Count);
        var options = new EmbeddingGenerationOptions { Dimensions = _dimensions };

        for (int i = 0; i < texts.Count; i += _batchSize)
        {
            var batch = texts.Skip(i).Take(_batchSize).Select(t => t.Trim()).ToList();
            _logger.LogDebug("Embedding batch {S}–{E}/{T}", i, i + batch.Count, texts.Count);

            try
            {
                var result = await _client.GenerateEmbeddingsAsync(batch, options, ct);
                _tokens.AddEmbeddingTokens(result.Value.Usage.InputTokenCount);
                all.AddRange(result.Value.Select(e => e.ToFloats().ToArray()));
                _logger.LogDebug("Batch complete ({Tokens} tokens)", result.Value.Usage.InputTokenCount);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to generate embeddings for batch {S}-{E}: {Message}", 
                    i, i + batch.Count, ex.Message);
                throw;
            }
        }

        _logger.LogDebug("Generated {Count} embeddings total", all.Count);
        return [.. all];
    }
}
