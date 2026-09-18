"""Terminal Chat - Main interactive chat interface like Claude AI."""

import sys
import time
import logging
import threading
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style as PromptStyle

from backend.model_manager import ModelManager
from backend.inference import InferenceEngine
from backend.tokenizer_utils import TokenizerManager
from terminal.formatter import OutputFormatter
from terminal.commands import CommandHandler
from terminal.history import ChatHistory

logger = logging.getLogger(__name__)


# Prompt toolkit styling
PROMPT_STYLE = PromptStyle.from_dict({
    "prompt": "bold green",
})


class TerminalChat:
    """
    Main terminal chat interface that provides a Claude AI-like
    interactive experience with streaming, markdown rendering,
    and rich command support.
    """

    def __init__(
        self,
        model_name: str = "gpt2-medium",
        model_type: str = "causal",
        device: str = "auto",
        precision: str = "fp32",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        max_new_tokens: int = 512,
        history_dir: str = "./chat_history",
        history_size: int = 20,
        show_stats: bool = True,
    ):
        # Configuration
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.precision = precision
        self.system_prompt = system_prompt or (
            "You are a helpful, harmless, and honest AI assistant. "
            "You provide clear, well-structured answers. When writing code, "
            "you use proper formatting. You can reason step-by-step when "
            "solving complex problems."
        )
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.show_stats = show_stats

        # Components
        self.formatter = OutputFormatter()
        self.history = ChatHistory(history_dir=history_dir, max_turns=history_size)
        self.model_manager: Optional[ModelManager] = None
        self.tokenizer_manager: Optional[TokenizerManager] = None
        self.inference_engine: Optional[InferenceEngine] = None
        self.command_handler: Optional[CommandHandler] = None

        # Prompt session for input
        self.prompt_session = PromptSession(
            history=FileHistory(".nexus_llm_history"),
            auto_suggest=AutoSuggestFromHistory(),
            style=PROMPT_STYLE,
        )

        self._running = False

    def initialize(self) -> None:
        """Initialize all components and load the model."""
        self.formatter.print_info("Initializing Nexus-LLM...")

        # Initialize model
        self.formatter.print_info(f"Loading model: {self.model_name}")
        self.model_manager = ModelManager(
            model_name=self.model_name,
            model_type=self.model_type,
            device=self.device,
            precision=self.precision,
        )
        self.model_manager.load_model()

        # Initialize tokenizer manager
        self.tokenizer_manager = TokenizerManager(
            self.model_manager.tokenizer,
            self.model_manager.model_name,
        )

        # Initialize inference engine
        self.inference_engine = InferenceEngine(
            self.model_manager, self.tokenizer_manager
        )

        # Initialize command handler
        self.command_handler = CommandHandler(
            formatter=self.formatter,
            history=self.history,
            inference_engine=self.inference_engine,
            model_manager=self.model_manager,
        )

        # Show model info
        info = self.model_manager.model_info
        self.formatter.print_success(
            f"Model loaded: {info.get('name', 'unknown')} "
            f"({info.get('num_parameters_billions', 0)}B parameters) "
            f"on {info.get('device', 'unknown')}"
        )

    def run(self) -> None:
        """Main chat loop."""
        self._running = True

        # Show welcome banner
        self.formatter.print_welcome()

        # Initialize
        try:
            self.initialize()
        except Exception as e:
            self.formatter.print_error(f"Failed to initialize: {e}")
            self.formatter.print_info("Make sure you have the required packages installed:")
            self.formatter.print_info("  pip install torch transformers")
            return

        # Main loop
        while self._running:
            try:
                # Get user input
                user_input = self.prompt_session.prompt(
                    "You> ",
                    multiline=False,
                )

                if not user_input or not user_input.strip():
                    continue

                user_input = user_input.strip()

                # Handle commands
                if self.command_handler and self.command_handler.is_command(user_input):
                    self.command_handler.execute(user_input)
                    if self.command_handler.should_exit:
                        self._running = False
                        break
                    continue

                # Handle multiline input (ending with \)
                while user_input.endswith("\\"):
                    user_input = user_input[:-1]
                    continuation = self.prompt_session.prompt("  ... ")
                    user_input += "\n" + continuation

                # Display user message
                self.formatter.print_user_message(user_input)

                # Add to history
                self.history.add_message("user", user_input)

                # Generate response
                self._generate_response()

            except KeyboardInterrupt:
                self.formatter.print_info("\nUse /quit to exit.")
                continue
            except EOFError:
                self.formatter.print_info("\nGoodbye!")
                self._running = False
                break
            except Exception as e:
                self.formatter.print_error(f"Unexpected error: {e}")
                logger.error(f"Chat loop error: {e}", exc_info=True)

        # Cleanup
        self._cleanup()

    def _generate_response(self) -> None:
        """Generate and display an assistant response with streaming."""
        if not self.inference_engine:
            self.formatter.print_error("Inference engine not initialized.")
            return

        # Show thinking indicator
        self.formatter.print_thinking()

        start_time = time.time()
        full_response = ""
        token_count = 0

        try:
            # Try streaming generation
            messages = self.history.get_messages(include_system=False)

            # Remove the last message (we just added the user message)
            # and format the conversation
            for token in self.inference_engine.chat_stream(
                messages=messages,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_new_tokens=self.max_new_tokens,
            ):
                if not self._running:
                    break
                full_response += token
                token_count += 1
                self.formatter.print_streaming_token(token)

        except Exception as e:
            # Fallback to non-streaming
            logger.warning(f"Streaming failed, falling back to batch: {e}")
            try:
                messages = self.history.get_messages(include_system=False)
                result = self.inference_engine.chat(
                    messages=messages,
                    system_prompt=self.system_prompt,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    max_new_tokens=self.max_new_tokens,
                )
                full_response = result["text"]
                token_count = result["output_tokens"]
            except Exception as e2:
                self.formatter.print_error(f"Generation failed: {e2}")
                return

        generation_time = time.time() - start_time
        tokens_per_second = token_count / generation_time if generation_time > 0 else 0

        # Display formatted response
        if full_response.strip():
            self.formatter.print_assistant_response(full_response)

            # Show stats
            if self.show_stats:
                input_tokens = self.model_manager.count_tokens(
                    self.tokenizer_manager.format_conversation(
                        self.history.get_messages(include_system=False),
                        system_prompt=self.system_prompt,
                    )
                )
                self.formatter.print_stats(
                    input_tokens=input_tokens,
                    output_tokens=token_count,
                    generation_time=generation_time,
                    tokens_per_second=tokens_per_second,
                )

        # Add to history
        self.history.add_message(
            "assistant",
            full_response,
            token_count=token_count,
            generation_time=generation_time,
        )

    def _cleanup(self) -> None:
        """Cleanup resources before exiting."""
        self.formatter.print_info("Cleaning up...")

        # Save session
        self.history.save_session()

        # Unload model
        if self.model_manager:
            self.model_manager.unload_model()

        self.formatter.print_info("Session saved. Goodbye!")

    def stop(self) -> None:
        """Stop the chat loop."""
        self._running = False
        if self.inference_engine:
            self.inference_engine.stop_generation()
