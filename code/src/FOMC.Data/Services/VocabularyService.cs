using System.Text.Json;
using System.Text.RegularExpressions;
using FOMC.Shared.Constants;
using Microsoft.Extensions.Logging;

namespace FOMC.Data.Services;

/// <summary>
/// Manages vocabulary mapping (term → integer ID) for sparse vector representation.
/// The vocabulary is built incrementally during ingestion and persisted to disk.
/// 
/// Sparse vectors in Qdrant require integer indices, so we need this mapping:
///   "inflation" → 142
///   "rates"     → 891
///   "federal"   → 2034
/// </summary>
public partial class VocabularyService
{
    private readonly ILogger<VocabularyService> _logger;
    private readonly string _vocabPath;
    
    // Term → ID mapping (grows during ingestion)
    private readonly Dictionary<string, uint> _termToId = new(StringComparer.OrdinalIgnoreCase);
    
    // ID → Term mapping (for debugging/inspection)
    private readonly Dictionary<uint, string> _idToTerm = new();
    
    // Document frequency: term → number of documents containing it
    private readonly Dictionary<string, int> _documentFrequency = new();
    
    // Total document count for IDF calculation
    private int _documentCount;
    
    // Next available ID
    private uint _nextId = 1;
    
    // Lock for thread-safe updates
    private readonly object _lock = new();

    public VocabularyService(ILogger<VocabularyService> logger, string vocabPath)
    {
        _logger = logger;
        _vocabPath = vocabPath;
    }

    /// <summary>
    /// Load vocabulary from disk if it exists.
    /// </summary>
    public async Task<bool> LoadAsync(CancellationToken ct = default)
    {
        if (!File.Exists(_vocabPath))
        {
            _logger.LogDebug("No vocabulary file found at {Path}, starting fresh", _vocabPath);
            return false;
        }

        try
        {
            var json = await File.ReadAllTextAsync(_vocabPath, ct);
            var data = JsonSerializer.Deserialize<VocabularyData>(json);
            
            if (data != null)
            {
                lock (_lock)
                {
                    _termToId.Clear();
                    _idToTerm.Clear();
                    _documentFrequency.Clear();
                    
                    foreach (var (term, id) in data.Terms)
                    {
                        _termToId[term] = id;
                        _idToTerm[id] = term;
                    }
                    
                    foreach (var (term, df) in data.DocumentFrequency)
                    {
                        _documentFrequency[term] = df;
                    }
                    
                    _documentCount = data.DocumentCount;
                    _nextId = data.NextId;
                }
                
                _logger.LogInformation(
                    "Loaded vocabulary: {Terms} terms, {Docs} documents",
                    _termToId.Count, _documentCount);
                return true;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to load vocabulary from {Path}", _vocabPath);
        }

        return false;
    }

    /// <summary>
    /// Save vocabulary to disk.
    /// </summary>
    public async Task SaveAsync(CancellationToken ct = default)
    {
        var dir = Path.GetDirectoryName(_vocabPath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            Directory.CreateDirectory(dir);

        VocabularyData data;
        lock (_lock)
        {
            data = new VocabularyData
            {
                Terms = new Dictionary<string, uint>(_termToId),
                DocumentFrequency = new Dictionary<string, int>(_documentFrequency),
                DocumentCount = _documentCount,
                NextId = _nextId,
                UpdatedAt = DateTime.UtcNow
            };
        }

        var json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = false });
        await File.WriteAllTextAsync(_vocabPath, json, ct);
        
        _logger.LogDebug("Saved vocabulary: {Terms} terms", data.Terms.Count);
    }

    /// <summary>
    /// Tokenize text and compute sparse vector representation.
    /// Returns (indices, values) for Qdrant sparse vector.
    /// </summary>
    public (uint[] Indices, float[] Values) ComputeSparseVector(string text, bool updateVocab = false)
    {
        var tokens = Tokenize(text);
        if (tokens.Count == 0)
            return ([], []);

        // Count term frequencies in this document
        var termFreqs = tokens
            .GroupBy(t => t)
            .ToDictionary(g => g.Key, g => g.Count());

        var indices = new List<uint>();
        var values = new List<float>();

        lock (_lock)
        {
            // Update document frequency if we're indexing
            if (updateVocab)
            {
                _documentCount++;
                foreach (var term in termFreqs.Keys)
                {
                    _documentFrequency[term] = _documentFrequency.GetValueOrDefault(term, 0) + 1;
                }
            }

            foreach (var (term, tf) in termFreqs)
            {
                uint termId;
                
                if (_termToId.TryGetValue(term, out var existingId))
                {
                    termId = existingId;
                }
                else if (updateVocab)
                {
                    // Add new term to vocabulary
                    termId = _nextId++;
                    _termToId[term] = termId;
                    _idToTerm[termId] = term;
                }
                else
                {
                    // Term not in vocabulary, skip (query-time only)
                    continue;
                }

                // Compute BM25-style weight
                // TF component with saturation: tf / (tf + k1)
                const float k1 = 1.2f;
                float tfWeight = tf / (tf + k1);
                
                // IDF component: log((N - df + 0.5) / (df + 0.5) + 1)
                float idf = 1.0f; // Default if no DF info
                if (_documentFrequency.TryGetValue(term, out var df) && _documentCount > 0)
                {
                    idf = (float)Math.Log((_documentCount - df + 0.5) / (df + 0.5) + 1);
                }

                float weight = tfWeight * idf;
                
                indices.Add(termId);
                values.Add(weight);
            }
        }

        return (indices.ToArray(), values.ToArray());
    }

    /// <summary>
    /// Get term ID for a given term (for querying).
    /// </summary>
    public uint? GetTermId(string term)
    {
        lock (_lock)
        {
            return _termToId.TryGetValue(term, out var id) ? id : null;
        }
    }

    /// <summary>
    /// Get vocabulary statistics.
    /// </summary>
    public (int TermCount, int DocCount) GetStats()
    {
        lock (_lock)
        {
            return (_termToId.Count, _documentCount);
        }
    }

    /// <summary>
    /// Tokenize text into lowercase terms, removing stop words.
    /// </summary>
    private static List<string> Tokenize(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return [];

        return WordPattern().Matches(text.ToLowerInvariant())
            .Select(m => m.Value)
            .Where(t => t.Length >= 2)
            .Where(t => !StopWords.IsStopWord(t))
            .Where(t => !IsNumeric(t))
            .ToList();
    }

    private static bool IsNumeric(string s) => s.All(char.IsDigit);

    [GeneratedRegex(@"\w+", RegexOptions.Compiled)]
    private static partial Regex WordPattern();

    /// <summary>
    /// Serialization model for vocabulary persistence.
    /// </summary>
    private class VocabularyData
    {
        public Dictionary<string, uint> Terms { get; set; } = new();
        public Dictionary<string, int> DocumentFrequency { get; set; } = new();
        public int DocumentCount { get; set; }
        public uint NextId { get; set; }
        public DateTime UpdatedAt { get; set; }
    }
}
