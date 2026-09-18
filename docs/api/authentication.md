# API Authentication

Nexus-LLM supports two authentication methods: API keys and JWT tokens.

## API Key Authentication

API keys are the simplest authentication method. Include your key in the `X-API-Key` header with every request.

### Creating API Keys

Use the CLI to generate a new API key:

```bash
nexus-llm auth create-key --user-id dev_001 --role developer
```

Or generate programmatically:

```python
from nexus_llm.auth import APIKeyAuth

auth = APIKeyAuth(key_prefix="nxs-", key_length=32)
api_key = auth.generate_key(user_id="dev_001")
print(f"API Key: {api_key}")  # nxs-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> **Important**: Save the API key immediately. It cannot be retrieved later because only the hashed version is stored.

### Using API Keys

```bash
curl -H "X-API-Key: nxs-your-api-key-here" \
     -H "Content-Type: application/json" \
     -d '{"model": "nexus-7b-chat", "messages": [...]}' \
     http://localhost:8000/api/v1/chat
```

```python
from nexus_llm.client import NexusClient

client = NexusClient(
    base_url="http://localhost:8000",
    api_key="nxs-your-api-key-here",
)

response = client.chat(
    model="nexus-7b-chat",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### API Key Security

- Keys are stored hashed (SHA-256) on the server.
- Keys have a configurable prefix for easy identification.
- Keys can be revoked at any time.
- Keys are associated with a specific role and rate limit policy.

### Revoking API Keys

```bash
nexus-llm auth revoke-key --key-id key_abc123
```

## JWT Authentication

JWT tokens provide time-limited access with refresh capabilities. They are recommended for client applications that need to maintain sessions.

### Obtaining a Token

```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
     -H "Content-Type: application/json" \
     -d '{"user_id": "dev_001", "api_key": "nxs-your-api-key-here"}'
```

Response:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Using JWT Tokens

Include the access token in the `Authorization` header:

```bash
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
     -H "Content-Type: application/json" \
     -d '{"model": "nexus-7b-chat", "messages": [...]}' \
     http://localhost:8000/api/v1/chat
```

### Refreshing Tokens

Access tokens expire after a configurable time (default: 1 hour). Use the refresh token to get a new access token:

```bash
curl -X POST http://localhost:8000/api/v1/auth/refresh \
     -H "Content-Type: application/json" \
     -d '{"refresh_token": "eyJhbGciOiJIUzI1NiIs..."}'
```

## Roles and Permissions

| Role | MODEL_LIST | MODEL_INFER | MODEL_MANAGE | USER_MANAGE | SYSTEM_CONFIG | VIEW_METRICS |
|------|:----------:|:-----------:|:------------:|:-----------:|:-------------:|:------------:|
| admin | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| developer | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| readonly | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |

### Permission Details

- **MODEL_LIST**: List available models
- **MODEL_INFER**: Run inference (chat, embeddings)
- **MODEL_MANAGE**: Load, unload, and configure models
- **USER_MANAGE**: Create, modify, and delete users
- **SYSTEM_CONFIG**: Modify server configuration
- **VIEW_METRICS**: Access monitoring metrics

## Rate Limiting

Each user has a rate limit policy that controls:

| Limit | Default (readonly) | Default (developer) | Default (admin) |
|-------|--------------------|--------------------|-----------------|
| Requests/minute | 30 | 100 | 1000 |
| Tokens/day | 100,000 | 500,000 | 1,000,000 |
| Concurrent requests | 5 | 20 | 100 |

When a rate limit is exceeded, the API returns a `429 Too Many Requests` response:

```json
{
  "error": {
    "type": "rate_limit_error",
    "message": "Rate limit exceeded: 30 requests per minute",
    "retry_after_seconds": 45
  }
}
```

## Configuration

### Server Configuration

```yaml
auth:
  enabled: true
  providers:
    - api_key
    - jwt
  jwt:
    secret_key: "${JWT_SECRET}"    # Use environment variable
    algorithm: HS256
    access_token_expire_minutes: 60
    refresh_token_expire_days: 7
  api_key:
    prefix: "nxs-"
    key_length: 32
    hash_keys: true
  exempt_endpoints:
    - /health
    - /api/v1/auth/token
```

### Best Practices

1. **Never commit secrets** to version control. Use environment variables.
2. **Use HTTPS** in production to protect tokens in transit.
3. **Rotate API keys** regularly (at least every 90 days).
4. **Use the minimum role** necessary for each user.
5. **Set appropriate rate limits** to prevent abuse.
6. **Monitor authentication logs** for suspicious activity.
7. **Use short-lived access tokens** with refresh token rotation.
