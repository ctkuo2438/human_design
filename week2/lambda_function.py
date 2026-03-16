import json
import os
import urllib.request

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

EXTRACTION_PROMPT = """Analyze this Human Design BodyGraph chart image. Extract the following information and return ONLY a JSON object with no other text:

{
  "type": "The Human Design Type (Generator, Manifesting Generator, Projector, Manifestor, or Reflector)",
  "authority": "The Inner Authority (e.g., Emotional/Solar Plexus, Sacral, Splenic, etc.)",
  "profile": "The Profile numbers (e.g., 4/6, 1/3, etc.)",
  "strategy": "The Strategy (e.g., To Respond, To Wait for the Invitation, To Inform, To Wait a Lunar Cycle)",
  "definition": "The Definition type (Single, Split, Triple Split, Quadruple Split, No Definition)",
  "defined_centers": ["List of all colored/defined centers"],
  "undefined_centers": ["List of all white/undefined centers"],
  "active_channels": ["List of active channels as gate-gate pairs"]
}

IMPORTANT RULES:
- Colored/shaded centers (any color: red, orange, yellow, brown, purple) are DEFINED
- White/empty centers are UNDEFINED
- The 9 centers are: Head, Ajna, Throat, G/Self, Heart/Will/Ego, Sacral, Solar Plexus/Emotional, Spleen, Root
- Channels are the lines connecting two centers - if a channel is fully colored, it is active
- Look at the Design (left) and Personality (right) columns for gate numbers
- Return ONLY valid JSON, no markdown, no explanation"""


def call_anthropic(image_base64, media_type="image/png"):
    """Call Anthropic API using urllib (no external dependencies)"""
    
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1500,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": EXTRACTION_PROMPT
                    }
                ],
            }
        ],
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    
    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    with urllib.request.urlopen(req, timeout=25) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    
    return result["content"][0]["text"]


def parse_json_response(text):
    """Parse JSON from LLM response, handling markdown code blocks"""
    import re
    # remove markdown code blocks if present
    cleaned = re.sub(r'```json\s*', '', text)
    cleaned = re.sub(r'```\s*', '', cleaned)
    return json.loads(cleaned.strip())

# lambda entry point
def lambda_handler(event, context):
    try:
        # parse request body, API Gateway for phase4
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])
        else:
            body = event.get("body") or event
        
        image_base64 = body.get("image")
        media_type = body.get("media_type", "image/png")
        
        if not image_base64:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Missing 'image' field (base64 encoded)"})
            }
        
        # call Anthropic Vision API to analyze image
        raw_response = call_anthropic(image_base64, media_type)
        
        # parse JSON and return text
        chart_data = parse_json_response(raw_response)
        
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "success": True,
                "chart_data": chart_data
            })
        }
    
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "success": False,
                "error": str(e)
            })
        }