# Research notes

## UI verification via Playwright MCP

The Streamlit UI was verified end to end with Playwright MCP browser tools
(navigate → wait → type → wait → screenshot). Key step:

```
playwright - Wait for(text: "Knowledge-Base Chatbot") (MCP)
Wait for text to appear or disappear or a specified time to pass
```

Streamlit renders asynchronously after the initial page load, so automation
must wait for app text (the page title above, then the expected answer text,
e.g. "Eating Out, Smarter Choices") rather than relying on navigation events.
The same wait-for-text step is used to confirm each chat response before
taking the verification screenshot.

