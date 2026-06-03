#!/usr/bin/env python3
"""
Authentication Example - Nexus-LLM
====================================
Demonstrates how to set up API key authentication, token-based
auth, and access control for Nexus-LLM server deployments.
"""

from nexus_llm.server import NexusServer, ServerConfig
from nexus_llm.auth import (
    AuthManager,
    APIKeyAuth,
    JWTAuth,
    Role,
    Permission,
    RateLimitPolicy,
    User,
)


def main():
    # --- Define roles and permissions ---
    admin_role = Role(
        name="admin",
        permissions=[
            Permission.MODEL_LIST,
            Permission.MODEL_INFER,
            Permission.MODEL_MANAGE,
            Permission.USER_MANAGE,
            Permission.SYSTEM_CONFIG,
            Permission.VIEW_METRICS,
        ],
    )

    developer_role = Role(
        name="developer",
        permissions=[
            Permission.MODEL_LIST,
            Permission.MODEL_INFER,
            Permission.VIEW_METRICS,
        ],
    )

    readonly_role = Role(
        name="readonly",
        permissions=[
            Permission.MODEL_LIST,
            Permission.MODEL_INFER,
        ],
    )

    # --- Create users ---
    users = [
        User(
            user_id="admin_001",
            name="System Admin",
            email="admin@example.com",
            role=admin_role,
            rate_limit=RateLimitPolicy(
                requests_per_minute=1000,
                tokens_per_day=1_000_000,
            ),
        ),
        User(
            user_id="dev_001",
            name="Developer Alice",
            email="alice@example.com",
            role=developer_role,
            rate_limit=RateLimitPolicy(
                requests_per_minute=100,
                tokens_per_day=500_000,
            ),
        ),
        User(
            user_id="user_001",
            name="Bob Client",
            email="bob@client.com",
            role=readonly_role,
            rate_limit=RateLimitPolicy(
                requests_per_minute=30,
                tokens_per_day=100_000,
            ),
        ),
    ]

    # --- Set up authentication ---
    auth_manager = AuthManager()

    # API Key authentication
    api_key_auth = APIKeyAuth(
        key_prefix="nxs-",           # API keys start with this prefix
        key_length=32,               # Length of the random part
        hash_keys=True,              # Store hashed keys for security
        key_file="./config/api_keys.json",
    )

    # Generate API keys for users
    for user in users:
        api_key = api_key_auth.generate_key(user.user_id)
        print(f"Generated API key for {user.name}: {api_key}")

    # JWT authentication
    jwt_auth = JWTAuth(
        secret_key="your-secret-key-change-in-production",
        algorithm="HS256",
        access_token_expire_minutes=60,
        refresh_token_expire_days=7,
        issuer="nexus-llm",
    )

    # Register auth providers
    auth_manager.register_provider("api_key", api_key_auth)
    auth_manager.register_provider("jwt", jwt_auth)

    # Register users
    for user in users:
        auth_manager.register_user(user)

    # --- Demonstrate JWT token flow ---
    print("\n" + "=" * 60)
    print("JWT Authentication Flow")
    print("=" * 60)

    # Authenticate and get tokens
    credentials = {"user_id": "dev_001", "email": "alice@example.com"}
    token_response = jwt_auth.authenticate(credentials)
    print(f"Access token: {token_response.access_token[:50]}...")
    print(f"Refresh token: {token_response.refresh_token[:50]}...")
    print(f"Expires in: {token_response.expires_in}s")

    # Validate a token
    claims = jwt_auth.validate_token(token_response.access_token)
    print(f"\nToken claims: {claims}")

    # Check permissions
    has_permission = auth_manager.check_permission(
        user_id="dev_001",
        permission=Permission.MODEL_INFER,
    )
    print(f"Developer has MODEL_INFER permission: {has_permission}")

    has_admin_perm = auth_manager.check_permission(
        user_id="dev_001",
        permission=Permission.SYSTEM_CONFIG,
    )
    print(f"Developer has SYSTEM_CONFIG permission: {has_admin_perm}")

    # --- Attach auth to server ---
    print("\n" + "=" * 60)
    print("Server with Authentication")
    print("=" * 60)

    server_config = ServerConfig(
        host="0.0.0.0",
        port=8000,
        auth_manager=auth_manager,
        require_auth=True,               # Enforce authentication on all endpoints
        exempt_endpoints=["/health"],     # No auth needed for health check
    )

    server = NexusServer(config=server_config)

    # Endpoints inherit auth requirements
    @server.endpoint("/api/v1/chat", methods=["POST"])
    @server.require_permission(Permission.MODEL_INFER)
    async def chat(request):
        """Authenticated chat endpoint."""
        user = request.auth_user
        # Rate limiting is handled automatically by the auth manager
        return {"response": "...", "user": user.name}

    @server.endpoint("/api/v1/admin/models", methods=["GET"])
    @server.require_permission(Permission.MODEL_MANAGE)
    async def list_models_admin(request):
        """Admin-only model management endpoint."""
        return {"models": []}

    print("Server configured with authentication.")
    print("Endpoints:")
    print("  POST /api/v1/chat      - Requires MODEL_INFER permission")
    print("  GET  /api/v1/admin/models - Requires MODEL_MANAGE permission")
    print("  GET  /health           - No authentication required")

    # --- Example API calls ---
    print("\n" + "=" * 60)
    print("Example API Calls")
    print("=" * 60)

    print("""
# Using API key:
curl -H "X-API-Key: nxs-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \\
     -H "Content-Type: application/json" \\
     -d '{"message": "Hello!"}' \\
     http://localhost:8000/api/v1/chat

# Using JWT token:
curl -H "Authorization: Bearer <access_token>" \\
     -H "Content-Type: application/json" \\
     -d '{"message": "Hello!"}' \\
     http://localhost:8000/api/v1/chat

# Get JWT token:
curl -X POST \\
     -H "Content-Type: application/json" \\
     -d '{"user_id": "dev_001", "email": "alice@example.com"}' \\
     http://localhost:8000/auth/token
""")


if __name__ == "__main__":
    main()
