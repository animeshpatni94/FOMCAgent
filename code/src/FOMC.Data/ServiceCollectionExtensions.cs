using FOMC.Data.Factories;
using FOMC.Data.Services;
using FOMC.Shared.Abstractions;
using FOMC.Shared.Configuration;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace FOMC.Data;

/// <summary>
/// Dependency Injection extensions for FOMC.Data services.
/// Centralizes service registration for clean startup code.
/// </summary>
public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Registers all FOMC.Data services with proper lifetimes.
    /// </summary>
    public static IServiceCollection AddFomcDataServices(
        this IServiceCollection services,
        IConfiguration configuration)
    {
        // ── Configuration Options (strongly-typed) ──────────────────────────
        services.Configure<AzureOpenAIOptions>(
            configuration.GetSection(AzureOpenAIOptions.SectionName));
        services.Configure<QdrantOptions>(
            configuration.GetSection(QdrantOptions.SectionName));
        services.Configure<QueryEnhancementOptions>(
            configuration.GetSection(QueryEnhancementOptions.SectionName));
        services.Configure<RerankingOptions>(
            configuration.GetSection(RerankingOptions.SectionName));
        services.Configure<PricingOptions>(
            configuration.GetSection(PricingOptions.SectionName));
        services.Configure<ChunkEnrichmentOptions>(
            configuration.GetSection(ChunkEnrichmentOptions.SectionName));

        // ── Factories ───────────────────────────────────────────────────────
        services.AddSingleton<IAzureOpenAIClientFactory, AzureOpenAIClientFactory>();

        // ── Core Services (Singleton for shared state like token tracking) ─
        services.AddSingleton<ITokenTracker, TokenTracker>();
        services.AddSingleton<ICostCalculator, CostCalculator>();

        // ── Data Services (Singleton - stateless, reusable) ─────────────────
        services.AddSingleton<IEmbeddingService, EmbeddingService>();
        services.AddSingleton<IVectorStore, QdrantService>();
        services.AddSingleton<IQueryEnhancer, QueryEnhancer>();
        services.AddSingleton<IReranker, Reranker>();

        return services;
    }

    /// <summary>
    /// Validates that required configuration sections are present.
    /// Call during startup to fail fast on misconfiguration.
    /// </summary>
    public static void ValidateFomcConfiguration(this IConfiguration configuration)
    {
        var azureSection = configuration.GetSection(AzureOpenAIOptions.SectionName);
        
        if (string.IsNullOrEmpty(azureSection["Endpoint"]))
            throw new InvalidOperationException(
                $"Configuration '{AzureOpenAIOptions.SectionName}:Endpoint' is required.");
        
        if (string.IsNullOrEmpty(azureSection["ApiKey"]))
            throw new InvalidOperationException(
                $"Configuration '{AzureOpenAIOptions.SectionName}:ApiKey' is required.");
    }
}
