# extract_web_article

Downloads and extracts the main text content from a web URL.

## Schema

```json
{
  "name": "extract_web_article",
  "description": "Fetches a URL and extracts the main article text, stripping out ads, navigation, and boilerplate. Ideal for reading blog posts, news articles, or documentation pages. Returns JSON containing the extracted text (up to max_chars limit) or detailed error info if the fetch fails.",
  "parameters": {
    "type": "object",
    "properties": {
      "url": {
        "type": "string",
        "description": "The full HTTP/HTTPS URL of the article to extract."
      },
      "max_chars": {
        "type": "integer",
        "description": "The maximum number of characters to return (default 12000) to prevent blowing up the LLM context.",
        "default": 12000
      }
    },
    "required": ["url"]
  }
}
```
