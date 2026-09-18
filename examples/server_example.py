#!/usr/bin/env python3
"""
Server Example - Nexus-LLM
============================
Demonstrates how to start and configure the Nexus-LLM API server.
"""

from nexus_llm.server import NexusServer, ServerConfig, MiddlewareConfig


def main():
    # Configure the server
    server_config = ServerConfig(
        host="0.0.0.0",
        port=8000,
        workers=4,
        cors_origins=["http://localhost:3000"],
        api_prefix="/api/v1",
        request_timeout=120,
        max_concurrent_requests=100,
    )

    # Configure middleware
    middleware_config = MiddlewareConfig(
        rate_limiting=True,
        rate_limit_requests=60,       # per minute
        rate_limit_period=60,
        authentication=True,
        api_key_header="X-API-Key",
        logging=True,
        log_level="info",
        request_id_tracking=True,
    )

    # Create the server instance
    server = NexusServer(
        config=server_config,
        middleware=middleware_config,
    )

    # Register models
    server.register_model(
        model_id="nexus-7b-chat",
        model_path="./models/nexus-7b-chat",
        device="auto",
        max_batch_size=8,
        max_concurrent=16,
    )
    server.register_model(
        model_id="nexus-3b-fast",
        model_path="./models/nexus-3b-chat",
        device="auto",
        max_batch_size=16,
        max_concurrent=32,
    )

    # Register custom endpoints
    @server.endpoint("/custom/summarize", methods=["POST"])
    async def summarize(request):
        """Custom endpoint for document summarization."""
        text = request.json["text"]
        max_length = request.json.get("max_length", 200)
        # ... summarization logic
        return {"summary": "..."}

    # Add startup/shutdown hooks
    @server.on_startup
    async def on_startup():
        print("Server starting up...")
        print("Loading models into memory...")

    @server.on_shutdown
    async def on_shutdown():
        print("Shutting down gracefully...")
        print("Unloading models...")

    # Start the server
    print(f"Starting Nexus-LLM server on {server_config.host}:{server_config.port}")
    server.run()

    # Alternatively, run with configuration file:
    # server = NexusServer.from_config("server_config.yaml")
    # server.run()


if __name__ == "__main__":
    main()
