namespace FOMC.Shared.Utilities;

/// <summary>
/// String manipulation utilities used across the solution.
/// </summary>
public static class StringHelpers
{
    /// <summary>
    /// Truncate a string to a maximum length, adding ellipsis if truncated.
    /// </summary>
    public static string Truncate(string text, int maxLength)
    {
        if (string.IsNullOrEmpty(text) || text.Length <= maxLength)
            return text;

        return maxLength <= 3 
            ? text[..maxLength] 
            : text[..(maxLength - 3)] + "...";
    }

    /// <summary>
    /// Truncate with a custom suffix.
    /// </summary>
    public static string Truncate(string text, int maxLength, string suffix)
    {
        if (string.IsNullOrEmpty(text) || text.Length <= maxLength)
            return text;

        var suffixLength = suffix.Length;
        return text[..(maxLength - suffixLength)] + suffix;
    }

    /// <summary>
    /// Join strings with a separator, filtering out null/empty values.
    /// </summary>
    public static string JoinNonEmpty(string separator, params string?[] values) =>
        string.Join(separator, values.Where(v => !string.IsNullOrWhiteSpace(v)));

    /// <summary>
    /// Normalize whitespace in text (collapse multiple spaces, trim).
    /// </summary>
    public static string NormalizeWhitespace(string text)
    {
        if (string.IsNullOrEmpty(text))
            return text;

        return System.Text.RegularExpressions.Regex
            .Replace(text.Trim(), @"\s+", " ");
    }

    // ══════════════════════════════════════════════════════════════════════════
    // SECURITY: Input Sanitization for Prompt Injection Protection
    // ══════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Sanitize user input to prevent prompt injection attacks.
    /// Removes or escapes common injection patterns while preserving query intent.
    /// </summary>
    public static string SanitizeForPrompt(string input)
    {
        if (string.IsNullOrEmpty(input))
            return input;

        var sanitized = input;

        // Remove common prompt injection markers
        var injectionPatterns = new[]
        {
            "ignore previous instructions",
            "ignore all instructions",
            "forget your instructions",
            "disregard the above",
            "new instructions:",
            "system:",
            "assistant:",
            "### instruction",
            "```system",
            "you are now",
            "act as",
            "pretend to be",
            "roleplay as",
            "jailbreak",
            "<|im_start|>",
            "<|im_end|>",
            "{{",
            "}}",
        };

        foreach (var pattern in injectionPatterns)
        {
            sanitized = System.Text.RegularExpressions.Regex.Replace(
                sanitized, 
                System.Text.RegularExpressions.Regex.Escape(pattern), 
                "[removed]", 
                System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        }

        // Limit length to prevent context overflow attacks
        const int MaxQueryLength = 2000;
        if (sanitized.Length > MaxQueryLength)
            sanitized = sanitized[..MaxQueryLength];

        // Normalize whitespace
        sanitized = NormalizeWhitespace(sanitized);

        return sanitized;
    }

    /// <summary>
    /// Check if input contains potential prompt injection attempts.
    /// Returns true if suspicious patterns detected.
    /// </summary>
    public static bool ContainsInjectionAttempt(string input)
    {
        if (string.IsNullOrEmpty(input))
            return false;

        var suspiciousPatterns = new[]
        {
            @"ignore\s+(previous|all|above|prior)\s+instructions",
            @"forget\s+(your|all|the)\s+instructions",
            @"new\s+instructions\s*:",
            @"system\s*:",
            @"you\s+are\s+now",
            @"act\s+as\s+a",
            @"pretend\s+to\s+be",
            @"<\|im_",
            @"\{\{.*\}\}",
            @"```\s*system",
        };

        var lowerInput = input.ToLowerInvariant();
        foreach (var pattern in suspiciousPatterns)
        {
            if (System.Text.RegularExpressions.Regex.IsMatch(lowerInput, pattern))
                return true;
        }

        return false;
    }
}
