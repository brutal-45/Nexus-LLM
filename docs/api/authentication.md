# Authentication

Nexus-LLM supports multiple authentication methods to secure your API endpoints.

## API Key Authentication

The primary authentication method is API key-based. Each request must include a valid API key in the `Authorization` header.

### Header Format

```bash
Authorization: Bearer YOUR_API_KEY
```

### Example

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer nxs_sk_abc123def456" \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "Hello"}]}'
```

### API Key Format

Nexus-LLM API keys follow the format:

```
nxs_sk_<random_32_hex_characters>
```

Example: `nxs_sk_a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6`

---

## Managing API Keys

### Creating API Keys

Use the CLI to create new API keys:

```bash
# Create a new API key
nexus-llm api-key create --name "my-app" --tier standard

# Create with specific permissions
nexus-llm api-key create \
  --name "read-only" \
  --tier free \
  --permissions chat,embeddings

# Create with expiration
nexus-llm api-key create \
  --name "temporary" \
  --expires-in 7d
```

### Listing API Keys

```bash
nexus-llm api-key list
```

Output:

```
ID              Name            Tier        Created             Expires     Status
────────────────────────────────────────────────────────────────────────────────────
nxs_sk_abc123   my-app          standard    2024-01-15 10:30    Never       active
nxs_sk_def456   read-only       free        2024-01-16 14:20    Never       active
nxs_sk_ghi789   temporary       free        2024-01-20 09:00    2024-01-27  active
```

### Revoking API Keys

```bash
nexus-llm api-key revoke nxs_sk_abc123
```

### Setting the Master API Key

The master API key is configured via the `NEXUS_API_KEY` environment variable or the `api_key` field in your configuration file:

```bash
# In .env
NEXUS_API_KEY=nxs_sk_your_master_key_here

# Or generate one
openssl rand -hex 32 | awk '{print "nxs_sk_"$1}'
```

The master API key has full administrative access and can manage other API keys.

---

## API Key Tiers

| Tier | Requests/min | Tokens/min | Requests/day | Models |
|---|---|---|---|---|
| `free` | 20 | 10,000 | 1,000 | Small models only |
| `standard` | 100 | 100,000 | 50,000 | All models |
| `professional` | 1,000 | 1,000,000 | Unlimited | All models + priority |
| `admin` | Unlimited | Unlimited | Unlimited | All + management APIs |

---

## JWT Authentication

For session-based authentication, Nexus-LLM supports JWT (JSON Web Tokens).

### Obtaining a JWT Token

```bash
curl -X POST http://localhost:8000/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "nxs_sk_abc123def456"
  }'
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### Using JWT Tokens

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "Hello"}]}'
```

### JWT Configuration

```bash
# In .env
NEXUS_JWT_SECRET=your-jwt-secret-key       # Required for JWT
NEXUS_JWT_EXPIRATION=86400                  # Token lifetime in seconds (default: 24h)
NEXUS_JWT_ALGORITHM=HS256                   # Signing algorithm
```

### Refreshing Tokens

```bash
curl -X POST http://localhost:8000/v1/auth/refresh \
  -H "Authorization: Bearer <current_jwt_token>"
```

---

## WebSocket Authentication

### Query Parameter

Pass the API key as a `token` query parameter when connecting:

```javascript
const ws = new WebSocket('ws://localhost:8000/v1/ws/chat?token=nxs_sk_abc123');
```

### Connection Message

Send an authentication message after connecting:

```javascript
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'auth',
    api_key: 'nxs_sk_abc123def456'
  }));
};
```

---

## Multi-Tenancy

Nexus-LLM supports project-level isolation for multi-tenant deployments.

### Project API Keys

```bash
# Create a project
nexus-llm project create --name "team-alpha" --tier professional

# Create a project-scoped API key
nexus-llm api-key create \
  --name "alpha-key" \
  --project "team-alpha" \
  --tier professional
```

### Project Headers

Include the project context in requests:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "X-Nexus-Project: team-alpha" \
  -H "Content-Type: application/json" \
  -d '...'
```

---

## Security Best Practices

1. **Never expose API keys in client-side code** — Use a backend proxy to handle API calls
2. **Rotate keys regularly** — Revoke and recreate keys every 90 days
3. **Use the minimum required tier** — Don't use admin keys for application access
4. **Set expiration dates** — Create temporary keys for short-lived access
5. **Use HTTPS in production** — Never transmit API keys over unencrypted connections
6. **Monitor usage** — Review API key usage logs regularly for anomalies
7. **Store secrets securely** — Use environment variables or a secrets manager, never commit to git

### Rate Limiting

All authenticated requests are rate-limited. When a rate limit is exceeded, the API returns a `429 Too Many Requests` response:

```json
{
  "error": {
    "message": "Rate limit exceeded: 100 requests/minute. Retry after 30 seconds.",
    "type": "rate_limit_error",
    "code": "rate_limit_exceeded"
  }
}
```

Rate limit headers are included in every response:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1703318460
Retry-After: 30
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `NEXUS_API_KEY` | — | Master API key |
| `NEXUS_JWT_SECRET` | — | JWT signing secret |
| `NEXUS_JWT_EXPIRATION` | `86400` | JWT token lifetime (seconds) |
| `NEXUS_JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `NEXUS_RATE_LIMIT` | `100/minute` | Default rate limit |
| `NEXUS_REQUIRE_AUTH` | `true` | Require authentication for all endpoints |
| `NEXUS_CORS_ORIGINS` | `["*"]` | Allowed CORS origins |
