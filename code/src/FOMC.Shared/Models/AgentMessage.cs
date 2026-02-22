namespace FOMC.Shared.Models;

public record AgentMessage
{
    public required string Role { get; init; }   // "user" | "assistant" | "tool"
    public required string Content { get; init; }
    public List<SearchResult> Citations { get; init; } = [];
}

public record AgentRequest
{
    public required string Question { get; init; }
    public List<AgentMessage> History { get; init; } = [];
}

public record AgentResponse
{
    public required string Answer { get; init; }
    public required List<SearchResult> Citations { get; init; }
    public required int ToolCallsUsed { get; init; }
    public string? Uncertainty { get; init; }
}
