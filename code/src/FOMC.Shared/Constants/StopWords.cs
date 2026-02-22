namespace FOMC.Shared.Constants;

/// <summary>
/// Common English stop words to filter during BM25 indexing and retrieval.
/// These words appear so frequently they provide no discriminative value.
/// </summary>
public static class StopWords
{
    /// <summary>
    /// Standard English stop words plus some domain-neutral additions.
    /// Excludes financial terms that might be meaningful in FOMC context.
    /// </summary>
    public static readonly HashSet<string> English = new(StringComparer.OrdinalIgnoreCase)
    {
        // Articles
        "a", "an", "the",
        
        // Pronouns
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", "she", "her", "hers", "herself",
        "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        
        // Verbs (common/auxiliary)
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing",
        "will", "would", "shall", "should", "may", "might", "must", "can", "could",
        
        // Prepositions
        "at", "by", "for", "from", "in", "into", "of", "on", "to", "with",
        "about", "above", "after", "against", "before", "below", "between",
        "during", "through", "under", "until", "upon", "within", "without",
        
        // Conjunctions
        "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
        "not", "only", "than", "when", "where", "while", "if", "then", "because",
        
        // Adverbs / Other common words
        "here", "there", "all", "any", "each", "every", "no", "some",
        "such", "own", "same", "other", "another", "more", "most", "very",
        "just", "also", "now", "how", "why", "well", "back", "even", "still",
        "too", "over", "out", "up", "down", "off", "again", "further",
        "once", "ever", "always", "never", "already", "often", "however",
        
        // Common verbs
        "get", "got", "getting", "go", "goes", "going", "went", "gone",
        "come", "came", "coming", "make", "made", "making", "take", "took", "taken",
        "see", "saw", "seen", "know", "knew", "known", "think", "thought",
        "say", "said", "tell", "told", "ask", "asked", "use", "used",
        
        // Numbers as words
        "one", "two", "first", "second",
        
        // Misc
        "like", "just", "also", "much", "many", "few", "less", "more",
        "new", "old", "good", "bad", "long", "short", "high", "low",
        "way", "thing", "things", "something", "anything", "nothing"
    };

    /// <summary>
    /// Check if a word is a stop word.
    /// </summary>
    public static bool IsStopWord(string word) => English.Contains(word);

    /// <summary>
    /// Filter stop words from a list of tokens.
    /// </summary>
    public static IEnumerable<string> RemoveStopWords(IEnumerable<string> tokens)
        => tokens.Where(t => !English.Contains(t));
}
