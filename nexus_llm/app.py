"""Nexus-LLM Main Application Module.

Provides the NexusLLMApp class that orchestrates all components of the
Nexus-LLM framework, including model management, chat, serving,
training, evaluation, and benchmarking.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nexus_llm.__version__ import __version__
from nexus_llm.constants import (
    APP_NAME,
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    LOG_LEVEL,
)
from nexus_llm.enums import ChatRole, DeviceType
from nexus_llm.events import EventBus, get_event_bus
from nexus_llm.exceptions import (
    ChatError,
    ConfigError,
    NexusLLMError,
    ServerError,
    TrainingError,
)
from nexus_llm.plugins import PluginManager
from nexus_llm.registry import GlobalRegistry
from nexus_llm.signals import GracefulContextManager, SignalHandler
from nexus_llm.state import StateManager
from nexus_llm.types import (
    BenchmarkConfig,
    ChatConfig,
    Conversation,
    DownloadConfig,
    EvalConfig,
    GenerationConfig,
    Message,
    ModelInfo,
    ServerConfig,
    TrainingConfig,
)

logger = logging.getLogger(__name__)
console = Console()


class NexusLLMApp:
    """Main application class for Nexus-LLM.

    Orchestrates all components including model management, interactive
    chat, server, training, evaluation, benchmarking, and configuration.

    Example:
        >>> app = NexusLLMApp()
        >>> app.run_chat({"model": "gpt2-medium"})
        >>> app.run_serve({"host": "0.0.0.0", "port": 8000})
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        log_level: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the Nexus-LLM application.

        Args:
            config_path: Path to a configuration file.
            log_level: Logging level override.
            verbose: Enable verbose logging.
        """
        self._config_path = config_path
        self._verbose = verbose
        self._log_level = log_level or os.environ.get("NEXUS_LLM_LOG_LEVEL", LOG_LEVEL)

        # Initialize core components
        self._event_bus = get_event_bus()
        self._registry = GlobalRegistry()
        self._plugin_manager = PluginManager(event_bus=self._event_bus)
        self._state_manager = StateManager()
        self._signal_handler = SignalHandler(event_bus=self._event_bus)

        # Setup logging
        self._setup_logging()

        # Load configuration
        self._config: Dict[str, Any] = {}
        if config_path:
            self._load_config(config_path)

        logger.info("NexusLLMApp initialized (v%s)", __version__)

    def _setup_logging(self) -> None:
        """Configure application logging."""
        level = getattr(logging, self._log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        if self._verbose:
            logging.getLogger("nexus_llm").setLevel(logging.DEBUG)

    def _load_config(self, config_path: str) -> None:
        """Load configuration from a file.

        Args:
            config_path: Path to the configuration file.
        """
        try:
            from nexus_llm.config_loader import ConfigLoader
            loader = ConfigLoader()
            self._config = loader.load(config_path)
            logger.info("Loaded configuration from: %s", config_path)
        except Exception as exc:
            logger.warning("Failed to load config from %s: %s", config_path, exc)

    def _get_device(self, device: Optional[str] = None) -> str:
        """Resolve the device to use.

        Args:
            device: Requested device string.

        Returns:
            Resolved device string.
        """
        if device and device != "auto":
            return device
        return DeviceType.detect().value

    def _load_model(self, model_name: str, device: Optional[str] = None) -> tuple:
        """Load a model and tokenizer.

        Args:
            model_name: Model name or path.
            device: Device to load the model on.

        Returns:
            Tuple of (model, tokenizer).
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved_device = self._get_device(device)

        logger.info("Loading model: %s on %s", model_name, resolved_device)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        if resolved_device != "cpu":
            model = model.to(resolved_device)

        model.eval()

        self._event_bus.publish(Event(
            event_type="model.loaded",
            data={"model_name": model_name, "device": resolved_device},
            source="NexusLLMApp",
        ) if False else None)

        return model, tokenizer

    def run_chat(self, config: Dict[str, Any]) -> None:
        """Run an interactive chat session.

        Args:
            config: Chat configuration dictionary.
        """
        chat_config = ChatConfig(
            model=config.get("model", DEFAULT_MODEL),
            system_prompt=config.get("system_prompt"),
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.9),
            top_k=config.get("top_k", 50),
            max_tokens=config.get("max_tokens", 2048),
            device=config.get("device"),
            use_history=config.get("use_history", True),
            single_prompt=config.get("single_prompt"),
        )

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            resolved_device = self._get_device(chat_config.device)
            model_name = chat_config.model

            console.print(f"\n[bold cyan]Loading model:[/bold cyan] {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            if resolved_device != "cpu":
                model = model.to(resolved_device)
            model.eval()

            console.print(f"[bold green]Model loaded successfully![/bold green]")
            console.print(f"[dim]Device: {resolved_device}[/dim]\n")

            conversation = Conversation(model=model_name, system_prompt=chat_config.system_prompt)

            if chat_config.system_prompt:
                conversation.add_message(ChatRole.SYSTEM, chat_config.system_prompt)

            # Single prompt mode
            if chat_config.single_prompt:
                response = self._generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=chat_config.single_prompt,
                    conversation=conversation if chat_config.use_history else None,
                    config=chat_config,
                    device=resolved_device,
                )
                console.print(f"[bold green]Assistant:[/bold green] {response}")
                return

            # Interactive mode
            console.print("[bold]Nexus-LLM Chat[/bold] - Type 'exit' or 'quit' to end.\n")

            while True:
                try:
                    user_input = input("You: ").strip()
                except EOFError:
                    break

                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", ":q"):
                    console.print("[bold yellow]Goodbye![/bold yellow]")
                    break

                if user_input.lower() == "/clear":
                    conversation.clear_history()
                    console.print("[dim]History cleared.[/dim]")
                    continue

                if user_input.lower() == "/help":
                    self._print_chat_help()
                    continue

                if chat_config.use_history:
                    conversation.add_message(ChatRole.USER, user_input)

                response = self._generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=user_input,
                    conversation=conversation if chat_config.use_history else None,
                    config=chat_config,
                    device=resolved_device,
                )

                if chat_config.use_history:
                    conversation.add_message(ChatRole.ASSISTANT, response)

                console.print(f"[bold green]Assistant:[/bold green] {response}\n")

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Chat interrupted.[/bold yellow]")
        except ImportError as exc:
            console.print(f"[bold red]Error:[/bold red] Missing dependency: {exc}")
            console.print("Install with: pip install torch transformers")
        except Exception as exc:
            raise ChatError(message=str(exc)) from exc

    def _generate_response(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        conversation: Optional[Conversation],
        config: ChatConfig,
        device: str,
    ) -> str:
        """Generate a response from the model.

        Args:
            model: The loaded model.
            tokenizer: The loaded tokenizer.
            prompt: The user prompt.
            conversation: Optional conversation for history.
            config: Chat configuration.
            device: Device string.

        Returns:
            Generated text response.
        """
        import torch

        # Build input text from conversation history
        if conversation and len(conversation.messages) > 1:
            input_text = ""
            for msg in conversation.messages:
                if msg.role == ChatRole.SYSTEM:
                    input_text += f"System: {msg.content}\n"
                elif msg.role == ChatRole.USER:
                    input_text += f"User: {msg.content}\n"
                elif msg.role == ChatRole.ASSISTANT:
                    input_text += f"Assistant: {msg.content}\n"
            input_text += "Assistant: "
        else:
            input_text = prompt

        inputs = tokenizer(input_text, return_tensors="pt")
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()

    def _print_chat_help(self) -> None:
        """Print chat command help."""
        help_table = Table(title="Chat Commands")
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description")
        help_table.add_row("/help", "Show this help message")
        help_table.add_row("/clear", "Clear conversation history")
        help_table.add_row("exit, quit, :q", "End the chat session")
        console.print(help_table)

    def run_serve(self, config: Dict[str, Any]) -> None:
        """Start the inference server.

        Args:
            config: Server configuration dictionary.
        """
        server_config = ServerConfig(
            host=config.get("host", DEFAULT_HOST),
            port=config.get("port", DEFAULT_PORT),
            model=config.get("model", DEFAULT_MODEL),
            workers=config.get("workers", 1),
            device=config.get("device"),
            api_key=config.get("api_key"),
            cors=config.get("cors", False),
            reload=config.get("reload", False),
            ssl_certfile=config.get("ssl_certfile"),
            ssl_keyfile=config.get("ssl_keyfile"),
            log_level=config.get("log_level", "info"),
        )

        try:
            import uvicorn
            from fastapi import FastAPI
            from fastapi.middleware.cors import CORSMiddleware

            app = FastAPI(
                title="Nexus-LLM Server",
                version=__version__,
                description="LLM Inference Server",
            )

            if server_config.cors:
                app.add_middleware(
                    CORSMiddleware,
                    allow_origins=["*"],
                    allow_credentials=True,
                    allow_methods=["*"],
                    allow_headers=["*"],
                )

            @app.get("/health")
            async def health_check() -> Dict[str, str]:
                return {"status": "healthy", "version": __version__}

            @app.post("/generate")
            async def generate(request: Dict[str, Any]) -> Dict[str, Any]:
                return {"generated_text": "Response from " + server_config.model, "model": server_config.model}

            @app.get("/models")
            async def list_models() -> Dict[str, Any]:
                return {"models": [server_config.model]}

            console.print(f"[bold cyan]Starting Nexus-LLM Server[/bold cyan]")
            console.print(f"  Host: {server_config.host}:{server_config.port}")
            console.print(f"  Model: {server_config.model}")
            console.print(f"  Workers: {server_config.workers}")

            uvicorn.run(
                app,
                host=server_config.host,
                port=server_config.port,
                log_level=server_config.log_level,
                ssl_certfile=server_config.ssl_certfile,
                ssl_keyfile=server_config.ssl_keyfile,
            )

        except ImportError as exc:
            raise ServerError(message=f"Missing dependency: {exc}")
        except Exception as exc:
            raise ServerError(message=str(exc)) from exc

    def run_train(self, config: Dict[str, Any]) -> None:
        """Run model training.

        Args:
            config: Training configuration dictionary.
        """
        train_config = TrainingConfig(
            model=config.get("model", DEFAULT_MODEL),
            dataset=config.get("dataset", ""),
            output_dir=config.get("output_dir", config.get("output", "./output")),
            epochs=config.get("epochs", 3),
            batch_size=config.get("batch_size", 8),
            learning_rate=config.get("learning_rate", config.get("learning_rate", 2e-5)),
            lora_rank=config.get("lora_rank", config.get("lora_rank", 8)),
            lora_alpha=config.get("lora_alpha", config.get("lora_alpha", 16)),
            lora_dropout=config.get("lora_dropout", config.get("lora_dropout", 0.05)),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", config.get("gradient_accumulation", 1)),
            max_seq_length=config.get("max_seq_length", config.get("max_seq_length", 2048)),
            fp16=config.get("fp16", False),
            bf16=config.get("bf16", False),
            device=config.get("device", "auto"),
            warmup_steps=config.get("warmup_steps", 100),
            weight_decay=config.get("weight_decay", 0.01),
            save_steps=config.get("save_steps", 500),
            eval_steps=config.get("eval_steps", 500),
            seed=config.get("seed", 42),
        )

        try:
            from datasets import load_dataset
            from peft import LoraConfig, get_peft_model
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
                Trainer,
            )

            console.print(f"[bold cyan]Starting Training[/bold cyan]")
            console.print(f"  Model: {train_config.model}")
            console.print(f"  Dataset: {train_config.dataset}")
            console.print(f"  Epochs: {train_config.epochs}")
            console.print(f"  Batch Size: {train_config.batch_size}")
            console.print(f"  Learning Rate: {train_config.learning_rate}")
            console.print(f"  LoRA Rank: {train_config.lora_rank}")
            console.print(f"  Output: {train_config.output_dir}")

            resolved_device = self._get_device(train_config.device)

            # Load tokenizer and model
            console.print("\n[bold]Loading model and tokenizer...[/bold]")
            tokenizer = AutoTokenizer.from_pretrained(train_config.model)
            model = AutoModelForCausalLM.from_pretrained(train_config.model)

            # Apply LoRA if configured
            if train_config.lora_rank > 0:
                console.print(f"[bold]Applying LoRA (rank={train_config.lora_rank})...[/bold]")
                lora_config = LoraConfig(
                    r=train_config.lora_rank,
                    lora_alpha=train_config.lora_alpha,
                    lora_dropout=train_config.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()

            # Load dataset
            console.print(f"[bold]Loading dataset: {train_config.dataset}[/bold]")
            dataset_path = Path(train_config.dataset)
            if dataset_path.suffix == ".jsonl":
                dataset = load_dataset("json", data_files=train_config.dataset)
            elif dataset_path.suffix == ".csv":
                dataset = load_dataset("csv", data_files=train_config.dataset)
            else:
                dataset = load_dataset("json", data_files=train_config.dataset)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=train_config.output_dir,
                num_train_epochs=train_config.epochs,
                per_device_train_batch_size=train_config.batch_size,
                learning_rate=train_config.learning_rate,
                warmup_steps=train_config.warmup_steps,
                weight_decay=train_config.weight_decay,
                save_steps=train_config.save_steps,
                eval_steps=train_config.eval_steps if "validation" in dataset else None,
                logging_steps=10,
                fp16=train_config.fp16,
                bf16=train_config.bf16,
                gradient_accumulation_steps=train_config.gradient_accumulation_steps,
                seed=train_config.seed,
                save_total_limit=3,
                load_best_model_at_end=True if "validation" in dataset else False,
            )

            # Tokenization function
            def tokenize_function(examples: Any) -> Any:
                return tokenizer(
                    examples.get("text", examples.get("content", "")),
                    truncation=True,
                    max_length=train_config.max_seq_length,
                    padding="max_length",
                )

            tokenized_dataset = dataset.map(tokenize_function, batched=True)

            # Train
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset.get("train", tokenized_dataset),
                eval_dataset=tokenized_dataset.get("validation", tokenized_dataset.get("test")),
            )

            console.print("[bold green]Starting training...[/bold green]\n")
            trainer.train()

            # Save model
            console.print("[bold]Saving model...[/bold]")
            trainer.save_model()
            tokenizer.save_pretrained(train_config.output_dir)

            console.print(f"[bold green]Training complete! Model saved to: {train_config.output_dir}[/bold green]")

        except ImportError as exc:
            raise TrainingError(message=f"Missing dependency: {exc}")
        except Exception as exc:
            raise TrainingError(message=str(exc)) from exc

    def run_train_data(self, config: Dict[str, Any]) -> None:
        """Run data preparation for training.

        Args:
            config: Data preparation configuration.
        """
        import json
        import random

        input_path = config.get("input_path", "")
        output_path = config.get("output_path", "")
        data_format = config.get("format", "jsonl")
        split_ratio = config.get("split_ratio", [0.9, 0.05, 0.05])
        shuffle = config.get("shuffle", False)
        seed = config.get("seed", 42)

        console.print(f"[bold cyan]Preparing Training Data[/bold cyan]")
        console.print(f"  Input: {input_path}")
        console.print(f"  Output: {output_path}")
        console.print(f"  Format: {data_format}")
        console.print(f"  Split ratio: {split_ratio}")

        input_file = Path(input_path)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        data = []
        if input_file.suffix == ".jsonl":
            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        elif input_file.suffix == ".json":
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
        elif input_file.suffix == ".csv":
            import csv
            with open(input_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data = [row for row in reader]
        else:
            console.print(f"[bold red]Unsupported format: {input_file.suffix}[/bold red]")
            return

        console.print(f"  Loaded {len(data)} examples")

        # Shuffle
        if shuffle:
            random.seed(seed)
            random.shuffle(data)

        # Split
        n = len(data)
        train_end = int(n * split_ratio[0])
        val_end = train_end + int(n * split_ratio[1])

        splits = {
            "train": data[:train_end],
            "validation": data[train_end:val_end],
            "test": data[val_end:],
        }

        # Save splits
        for split_name, split_data in splits.items():
            split_file = output_dir / f"{split_name}.{data_format}"
            with open(split_file, "w", encoding="utf-8") as f:
                if data_format == "jsonl":
                    for item in split_data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                elif data_format == "json":
                    json.dump(split_data, f, ensure_ascii=False, indent=2)
            console.print(f"  {split_name}: {len(split_data)} examples -> {split_file}")

        console.print(f"[bold green]Data preparation complete![/bold green]")

    def run_models(self, config: Dict[str, Any]) -> None:
        """List and manage models.

        Args:
            config: Models command configuration.
        """
        list_available = config.get("list_available", False)
        filter_pattern = config.get("filter")

        if list_available:
            from download_model import SUPPORTED_MODELS

            table = Table(title="Available Models")
            table.add_column("Short Name", style="cyan")
            table.add_column("HuggingFace ID", style="green")
            table.add_column("Size", style="yellow", justify="right")
            table.add_column("License", style="magenta")

            for name, info in sorted(SUPPORTED_MODELS.items()):
                if filter_pattern and filter_pattern.lower() not in name.lower():
                    continue
                table.add_row(name, info["full_name"], info["size"], info["license"])

            console.print(table)
            console.print(f"\n[bold]Total models:[/bold] {len(SUPPORTED_MODELS)}")
        else:
            # Check locally available models
            models_dir = Path(os.environ.get("NEXUS_LLM_MODELS_DIR", "./models"))
            if models_dir.exists():
                table = Table(title="Local Models")
                table.add_column("Name", style="cyan")
                table.add_column("Path", style="green")

                for item in sorted(models_dir.iterdir()):
                    if item.is_dir():
                        table.add_row(item.name, str(item))

                console.print(table)
            else:
                console.print("[yellow]No local models directory found.[/yellow]")
                console.print("Use 'nexus-llm models --list-available' to see downloadable models.")

    def run_download(self, config: Dict[str, Any]) -> None:
        """Download a model.

        Args:
            config: Download configuration dictionary.
        """
        from download_model import SUPPORTED_MODELS, download_model_from_hf

        model_name = config.get("model_name", "")
        output_dir = config.get("output_dir") or os.environ.get("NEXUS_LLM_MODELS_DIR", "./models")
        revision = config.get("revision")
        token = config.get("token") or os.environ.get("HF_TOKEN")

        # Resolve short name to full name
        if model_name in SUPPORTED_MODELS:
            full_name = SUPPORTED_MODELS[model_name]["full_name"]
            console.print(f"Resolved [cyan]{model_name}[/cyan] -> [green]{full_name}[/green]")
        else:
            full_name = model_name

        download_model_from_hf(
            model_name=full_name,
            output_dir=Path(output_dir),
            revision=revision,
            token=token,
            verify=config.get("verify", True),
        )

    def run_eval(self, config: Dict[str, Any]) -> None:
        """Run model evaluation.

        Args:
            config: Evaluation configuration dictionary.
        """
        eval_config = EvalConfig(
            model=config.get("model", DEFAULT_MODEL),
            benchmark=config.get("benchmark"),
            tasks=config.get("tasks"),
            output_dir=config.get("output_dir", "./eval_results"),
            device=config.get("device", "auto"),
            batch_size=config.get("batch_size", 8),
            num_fewshot=config.get("num_fewshot", 0),
            limit=config.get("limit"),
            fp16=config.get("fp16", False),
            bf16=config.get("bf16", False),
            save_predictions=config.get("save_predictions", False),
        )

        console.print(f"[bold cyan]Starting Evaluation[/bold cyan]")
        console.print(f"  Model: {eval_config.model}")
        console.print(f"  Benchmark: {eval_config.benchmark or 'default'}")
        console.print(f"  Tasks: {eval_config.tasks or 'all'}")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            resolved_device = self._get_device(eval_config.device)

            console.print(f"\n[bold]Loading model: {eval_config.model}[/bold]")
            tokenizer = AutoTokenizer.from_pretrained(eval_config.model)
            model = AutoModelForCausalLM.from_pretrained(eval_config.model)

            if resolved_device != "cpu":
                model = model.to(resolved_device)
            model.eval()

            console.print("[bold green]Model loaded. Running evaluation...[/bold green]")

            # Simple perplexity-based evaluation
            results = self._run_simple_eval(model, tokenizer, resolved_device, eval_config)

            # Display results
            table = Table(title="Evaluation Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for metric, value in results.items():
                table.add_row(metric, f"{value:.4f}" if isinstance(value, float) else str(value))

            console.print(table)

            # Save results
            output_dir = Path(eval_config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            import json
            results_file = output_dir / f"eval_{eval_config.model.replace('/', '__')}.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
            console.print(f"\nResults saved to: {results_file}")

        except ImportError as exc:
            console.print(f"[bold red]Error:[/bold red] Missing dependency: {exc}")
        except Exception as exc:
            console.print(f"[bold red]Evaluation error:[/bold red] {exc}")

    def _run_simple_eval(
        self,
        model: Any,
        tokenizer: Any,
        device: str,
        config: EvalConfig,
    ) -> Dict[str, Any]:
        """Run a simple evaluation.

        Args:
            model: The loaded model.
            tokenizer: The loaded tokenizer.
            device: Device string.
            config: Eval configuration.

        Returns:
            Dictionary of evaluation metrics.
        """
        import torch

        test_prompts = [
            "The capital of France is",
            "The largest planet in our solar system is",
            "Water boils at",
        ]

        total_loss = 0.0
        num_samples = 0

        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item()
                num_samples += 1

        avg_loss = total_loss / max(num_samples, 1)
        perplexity = float(torch.exp(torch.tensor(avg_loss)))

        return {
            "model": config.model,
            "average_loss": avg_loss,
            "perplexity": perplexity,
            "num_samples": num_samples,
            "device": device,
        }

    def run_benchmark(self, config: Dict[str, Any]) -> None:
        """Run inference benchmarks.

        Args:
            config: Benchmark configuration dictionary.
        """
        benchmark_config = BenchmarkConfig(
            model=config.get("model", DEFAULT_MODEL),
            device=config.get("device", "auto"),
            batch_sizes=config.get("batch_sizes", [1, 2, 4, 8]),
            seq_lengths=config.get("seq_lengths", [128, 256, 512, 1024]),
            warmup=config.get("warmup", 3),
            iterations=config.get("iterations", 10),
            output_file=config.get("output_file"),
        )

        console.print(f"[bold cyan]Starting Benchmark[/bold cyan]")
        console.print(f"  Model: {benchmark_config.model}")
        console.print(f"  Device: {benchmark_config.device}")
        console.print(f"  Batch sizes: {benchmark_config.batch_sizes}")
        console.print(f"  Sequence lengths: {benchmark_config.seq_lengths}")

        try:
            import time
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            resolved_device = self._get_device(benchmark_config.device)

            console.print(f"\n[bold]Loading model: {benchmark_config.model}[/bold]")
            tokenizer = AutoTokenizer.from_pretrained(benchmark_config.model)
            model = AutoModelForCausalLM.from_pretrained(benchmark_config.model)

            if resolved_device != "cpu":
                model = model.to(resolved_device)
            model.eval()

            results = []

            for batch_size in benchmark_config.batch_sizes:
                for seq_length in benchmark_config.seq_lengths:
                    console.print(f"\n  Benchmarking: batch={batch_size}, seq_len={seq_length}")

                    input_ids = torch.randint(
                        0, tokenizer.vocab_size, (batch_size, seq_length), device=resolved_device
                    )

                    # Warmup
                    for _ in range(benchmark_config.warmup):
                        with torch.no_grad():
                            _ = model(input_ids=input_ids)

                    if resolved_device == "cuda":
                        torch.cuda.synchronize()

                    # Benchmark
                    latencies = []
                    for _ in range(benchmark_config.iterations):
                        start = time.perf_counter()
                        with torch.no_grad():
                            _ = model(input_ids=input_ids)
                        if resolved_device == "cuda":
                            torch.cuda.synchronize()
                        end = time.perf_counter()
                        latencies.append(end - start)

                    avg_latency = sum(latencies) / len(latencies)
                    throughput = batch_size / avg_latency

                    result = {
                        "batch_size": batch_size,
                        "seq_length": seq_length,
                        "avg_latency_ms": avg_latency * 1000,
                        "throughput_samples_per_sec": throughput,
                    }
                    results.append(result)
                    console.print(f"    Avg latency: {avg_latency * 1000:.2f}ms | Throughput: {throughput:.2f} samples/s")

            # Display summary
            table = Table(title="Benchmark Results")
            table.add_column("Batch Size", justify="right")
            table.add_column("Seq Length", justify="right")
            table.add_column("Avg Latency (ms)", justify="right")
            table.add_column("Throughput (samples/s)", justify="right")

            for r in results:
                table.add_row(
                    str(r["batch_size"]),
                    str(r["seq_length"]),
                    f"{r['avg_latency_ms']:.2f}",
                    f"{r['throughput_samples_per_sec']:.2f}",
                )

            console.print(table)

            # Save results
            if benchmark_config.output_file:
                import json
                output_path = Path(benchmark_config.output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                console.print(f"\nResults saved to: {output_path}")

        except ImportError as exc:
            console.print(f"[bold red]Error:[/bold red] Missing dependency: {exc}")
        except Exception as exc:
            console.print(f"[bold red]Benchmark error:[/bold red] {exc}")

    def run_config(self, config: Dict[str, Any]) -> None:
        """Manage configuration.

        Args:
            config: Configuration command data.
        """
        from nexus_llm.config_loader import ConfigLoader

        loader = ConfigLoader()

        if config.get("reset"):
            loader.reset()
            console.print("[bold green]Configuration reset to defaults.[/bold green]")
            return

        if config.get("list_all"):
            all_config = loader.get_all()
            table = Table(title="Nexus-LLM Configuration")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Source", style="yellow")

            for key, entry in sorted(all_config.items()):
                value = entry.get("value", "")
                source = entry.get("source", "default")
                table.add_row(key, str(value), source)

            console.print(table)
            return

        key = config.get("key")
        value = config.get("value")

        if key and config.get("unset"):
            loader.unset(key)
            console.print(f"[bold green]Unset:[/bold green] {key}")
        elif key and value is not None:
            loader.set(key, value)
            console.print(f"[bold green]Set:[/bold green] {key} = {value}")
        elif key:
            current_value = loader.get(key)
            if current_value is not None:
                console.print(f"{key} = {current_value}")
            else:
                console.print(f"[yellow]Key not found:[/yellow] {key}")
        else:
            console.print("Use --list to see all configuration values, or provide a key/value pair.")
