#!/usr/bin/env python3
"""
Oh-Llama: A colorful CLI tool for interacting with Ollama models
"""

import argparse
import json
import os
import subprocess
import sys
import time
import requests
from typing import List, Dict, Any, Optional
import threading
import signal
import shutil

try:
    import colorama
    from colorama import Fore, Back, Style
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.styles import Style as PromptStyle
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "colorama", "prompt_toolkit", "rich", "requests"])
    import colorama
    from colorama import Fore, Back, Style
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.styles import Style as PromptStyle
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt

# Initialize colorama
colorama.init(autoreset=True)

# Initialize Rich console
console = Console()

# Constants
OLLAMA_API_BASE = "http://localhost:11434/api"
OLLAMA_MODELS_URL = "https://ollama.com/library"
HISTORY_FILE = os.path.expanduser("~/.oh_llama_history")

class OllamaService:
    """Service to interact with Ollama"""
    
    @staticmethod
    def is_ollama_running() -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{OLLAMA_API_BASE}/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    @staticmethod
    def start_ollama() -> bool:
        """Start the Ollama service"""
        console.print("[yellow]Starting Ollama service...[/]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[yellow]Starting Ollama service...[/]"),
            transient=True,
        ) as progress:
            progress.add_task("starting", total=None)
            
            # Start ollama in background
            subprocess.Popen(["ollama", "serve"], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
            
            # Wait for service to be available
            for _ in range(10):  # Try for 10 seconds
                if OllamaService.is_ollama_running():
                    console.print("[green]Ollama service started successfully![/]")
                    return True
                time.sleep(1)
        
        console.print("[red]Failed to start Ollama service![/]")
        return False
    
    @staticmethod
    def get_local_models() -> List[Dict[str, Any]]:
        """Get list of locally installed models"""
        try:
            response = requests.get(f"{OLLAMA_API_BASE}/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                # Sort models by name for consistent display
                return sorted(models, key=lambda x: x.get("name", ""))
            return []
        except requests.RequestException:
            console.print("[red]Error fetching local models[/]")
            return []
    
    @staticmethod
    def get_available_models() -> List[Dict[str, Any]]:
        """Get list of models available from Ollama library"""
        try:
            # This is a simplified approach - the actual Ollama library API might be different
            response = requests.get("https://ollama.ai/api/models")
            if response.status_code == 200:
                return response.json()
            
            # Fallback to hardcoded popular models if API fails
            return [
                {"name": "llama3", "description": "Meta's Llama 3 model"},
                {"name": "llama3:8b", "description": "Meta's Llama 3 8B parameters model"},
                {"name": "llama3:70b", "description": "Meta's Llama 3 70B parameters model"},
                {"name": "mistral", "description": "Mistral AI's base model"},
                {"name": "mixtral", "description": "Mistral AI's mixture of experts model"},
                {"name": "phi3", "description": "Microsoft's Phi-3 model"},
                {"name": "gemma", "description": "Google's Gemma model"},
                {"name": "codellama", "description": "Meta's Code Llama model"},
                {"name": "llava", "description": "Multimodal model with vision capabilities"},
                {"name": "orca-mini", "description": "Lightweight model for resource-constrained environments"}
            ]
        except requests.RequestException:
            console.print("[red]Error fetching available models[/]")
            return []
    
    @staticmethod
    def pull_model(model_name: str) -> bool:
        """Pull a model from Ollama library"""
        console.print(f"[yellow]Pulling model: {model_name}...[/]")
        
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in iter(process.stdout.readline, ''):
            console.print(f"[dim]{line.strip()}[/]")
        
        process.stdout.close()
        return process.wait() == 0
    
    @staticmethod
    def delete_model(model_name: str) -> bool:
        """Delete a model from local storage"""
        console.print(f"[yellow]Deleting model: {model_name}...[/]")
        
        try:
            process = subprocess.run(
                ["ollama", "rm", model_name],
                capture_output=True,
                text=True,
                check=True
            )
            console.print(f"[green]Model {model_name} deleted successfully![/]")
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error deleting model: {e.stderr}[/]")
            return False
    
    @staticmethod
    def generate_response(model: str, prompt: str) -> str:
        """Generate a response from the model without streaming"""
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False  # Don't stream, get complete response
        }
        
        try:
            response = requests.post(
                f"{OLLAMA_API_BASE}/generate", 
                headers=headers,
                json=data
            )
            
            if response.status_code != 200:
                return f"Error: Received status code {response.status_code}"
            
            response_data = response.json()
            return response_data.get("response", "")
            
        except requests.RequestException as e:
            return f"Error: {str(e)}"


class OhLlamaCLI:
    """Main CLI interface for Oh-Llama"""
    
    def __init__(self):
        self.ollama_service = OllamaService()
        self.selected_model = None
        self.console = Console()
        self.prompt_style = PromptStyle.from_dict({
            'prompt': '#00aa00 bold',
            'continuation': 'gray'
        })
        
        # Create prompt session with history
        self.session = PromptSession(
            history=FileHistory(HISTORY_FILE),
            auto_suggest=AutoSuggestFromHistory(),
            style=self.prompt_style
        )
    
    def print_header(self):
        """Print the application header"""
        terminal_width = shutil.get_terminal_size().columns
        
        header = """
   ____  _           _     _                         
  / __ \\| |__       | |   | | __ _ _ __ ___   __ _  
 | |  | | '_ \\ _____| |   | |/ _` | '_ ` _ \\ / _` | 
 | |__| | | | |_____| |___| | (_| | | | | | | (_| | 
  \\____/|_| |_|     |_____|_|\\__,_|_| |_| |_|\\__,_| 
                                                    
        """
        
        self.console.print(Panel(
            header, 
            title="[bold cyan]Oh-Llama CLI[/]",
            subtitle="[bold green]Interact with Ollama models[/]",
            width=min(80, terminal_width),
            style="cyan",
            border_style="cyan"
        ))
    
    def get_key(self):
        """Get a keypress including special keys"""
        import termios
        import tty
        
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            
            if ch == '\x1b':  # ESC sequence
                ch = sys.stdin.read(2)
                if ch == '[A':  # Up arrow
                    return "up"
                elif ch == '[B':  # Down arrow
                    return "down"
            elif ch == '\r':  # Enter
                return "enter"
            elif ch == '\x04':  # Ctrl+D
                return "ctrl_d"
            
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def select_model(self) -> Optional[str]:
        """Display model selection menu with arrow key navigation"""
        if not self.ollama_service.is_ollama_running():
            if not self.ollama_service.start_ollama():
                self.console.print("[red]Failed to start Ollama. Please make sure it's installed.[/]")
                return None
        
        local_models = self.ollama_service.get_local_models()
        
        if not local_models:
            self.console.print("[yellow]No local models found. Let's pull one from the library.[/]")
            return self.pull_new_model()
        
        # Display local models
        table = Table(title="[bold]Local Models[/]", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim")
        table.add_column("Model", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Modified", style="yellow")
        
        for i, model in enumerate(local_models, 1):
            # Format the modified date if available
            modified = model.get("modified", "")
            if modified:
                try:
                    # Convert timestamp to readable format
                    modified_time = time.strftime("%Y-%m-%d", time.localtime(int(modified)))
                except (ValueError, TypeError):
                    modified_time = "Unknown"
            else:
                modified_time = "Unknown"
                
            table.add_row(
                str(i),
                model.get("name", "Unknown"),
                f"{model.get('size', 0) / (1024*1024*1024):.1f} GB",
                modified_time
            )
        
        # Add option to pull new model and delete model
        options = [model.get("name", "Unknown") for model in local_models]
        options.append("Pull a new model")
        options.append("Delete a model")
        
        # Use rich for arrow key selection
        selected = 0
        
        def print_menu():
            self.console.clear()
            self.print_header()
            self.console.print(table)
            self.console.print("")
            
            for i, option in enumerate(options):
                if i == selected:
                    self.console.print(f"[bold green]> {option}[/]")
                else:
                    self.console.print(f"  {option}")
            
            # Command legend
            self.console.print("\n[dim]╭─ Commands ───────────────────────────────────────────╮[/]")
            self.console.print("[dim]│ ↑/↓: Navigate   Enter: Select   Ctrl+D: Exit        │[/]")
            self.console.print("[dim]╰────────────────────────────────────────────────────╯[/]")
        
        print_menu()
        
        # Handle key presses
        while True:
            try:
                key = self.get_key()
                
                if key == "up" and selected > 0:
                    selected -= 1
                    print_menu()
                elif key == "down" and selected < len(options) - 1:
                    selected += 1
                    print_menu()
                elif key == "enter":
                    if selected == len(options) - 2:  # Pull a new model
                        return self.pull_new_model()
                    elif selected == len(options) - 1:  # Delete a model
                        if self.delete_model(local_models):
                            # Refresh the model list after deletion
                            return self.select_model()
                        else:
                            print_menu()
                    else:
                        return local_models[selected]["name"]
                elif key == "ctrl_d":
                    self.console.print("\n[yellow]Exiting Oh-Llama...[/]")
                    sys.exit(0)
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Operation cancelled.[/]")
                return self.select_model()
    
    def pull_new_model(self) -> Optional[str]:
        """Menu to pull a new model with arrow key navigation"""
        available_models = self.ollama_service.get_available_models()
        
        if not available_models:
            self.console.print("[red]Failed to fetch available models.[/]")
            model_name = Prompt.ask("[yellow]Enter model name manually[/]")
            if self.ollama_service.pull_model(model_name):
                return model_name
            return None
        
        # Display available models
        table = Table(title="[bold]Available Models[/]", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim")
        table.add_column("Model", style="cyan")
        table.add_column("Description", style="green")
        
        for i, model in enumerate(available_models, 1):
            table.add_row(
                str(i),
                model.get("name", "Unknown"),
                model.get("description", "")
            )
        
        # Use arrow key selection
        selected = 0
        options = [model.get("name", "Unknown") for model in available_models]
        
        def print_menu():
            self.console.clear()
            self.print_header()
            self.console.print(table)
            self.console.print("")
            
            for i, option in enumerate(options):
                if i == selected:
                    self.console.print(f"[bold green]> {option}[/]")
                else:
                    self.console.print(f"  {option}")
            
            # Command legend
            self.console.print("\n[dim]╭─ Commands ───────────────────────────────────────────╮[/]")
            self.console.print("[dim]│ ↑/↓: Navigate   Enter: Select   Ctrl+D: Cancel      │[/]")
            self.console.print("[dim]╰────────────────────────────────────────────────────╯[/]")
        
        print_menu()
        
        # Handle key presses
        while True:
            try:
                key = self.get_key()
                
                if key == "up" and selected > 0:
                    selected -= 1
                    print_menu()
                elif key == "down" and selected < len(options) - 1:
                    selected += 1
                    print_menu()
                elif key == "enter":
                    model_name = available_models[selected]["name"]
                    if self.ollama_service.pull_model(model_name):
                        return model_name
                    return None
                elif key == "ctrl_d":
                    self.console.print("\n[yellow]Cancelled model pull.[/]")
                    time.sleep(1)
                    return self.select_model()
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Operation cancelled.[/]")
                time.sleep(1)
                return self.select_model()
    
    def delete_model(self, local_models: List[Dict[str, Any]]) -> bool:
        """Delete a model with confirmation"""
        self.console.clear()
        self.print_header()
        
        # Create a table of models to delete
        table = Table(title="[bold red]Delete Model[/]", show_header=True, header_style="bold red")
        table.add_column("#", style="dim")
        table.add_column("Model", style="cyan")
        table.add_column("Size", style="green")
        
        for i, model in enumerate(local_models, 1):
            table.add_row(
                str(i),
                model.get("name", "Unknown"),
                f"{model.get('size', 0) / (1024*1024*1024):.1f} GB"
            )
        
        self.console.print(table)
        self.console.print("\n[bold yellow]Select a model to delete or press Ctrl+D to cancel[/]")
        
        # Command legend
        self.console.print("\n[dim]╭─ Commands ───────────────────────────────────────────╮[/]")
        self.console.print("[dim]│ Enter number to select   Ctrl+D: Cancel             │[/]")
        self.console.print("[dim]╰────────────────────────────────────────────────────╯[/]")
        
        try:
            choice = Prompt.ask(
                "[bold red]Select model to delete[/]", 
                choices=[str(i) for i in range(1, len(local_models) + 1)],
                show_choices=False
            )
            
            choice = int(choice)
            if 1 <= choice <= len(local_models):
                model_name = local_models[choice - 1]["name"]
                
                # Confirm deletion
                self.console.print(f"\n[bold red]Are you sure you want to delete {model_name}?[/]")
                confirm = Prompt.ask(
                    "[bold red]Type 'yes' to confirm[/]",
                    choices=["yes", "no"],
                    default="no"
                )
                
                if confirm.lower() == "yes":
                    # Delete the model
                    if self.ollama_service.delete_model(model_name):
                        self.console.print(f"[green]Successfully deleted {model_name}[/]")
                        time.sleep(1)  # Give user time to see the message
                        return True
                    else:
                        self.console.print(f"[red]Failed to delete {model_name}[/]")
                        time.sleep(1)
                        return False
            
            return False
        except (EOFError, KeyboardInterrupt):
            self.console.print("\n[yellow]Deletion cancelled.[/]")
            time.sleep(1)
            return False
    
    def chat_loop(self, model: str):
        """Main chat loop with the selected model"""
        self.console.clear()
        self.print_header()
        self.console.print(f"\n[bold green]Chatting with model:[/] [cyan]{model}[/]")
        
        # Command legend
        self.console.print("\n[dim]╭─ Commands ───────────────────────────────────────────╮[/]")
        self.console.print("[dim]│ exit/quit: Exit chat   Ctrl+C: Cancel input          │[/]")
        self.console.print("[dim]│ Ctrl+D: Exit chat                                    │[/]")
        self.console.print("[dim]╰────────────────────────────────────────────────────╯[/]")
        
        while True:
            try:
                user_input = self.session.prompt(f"\n{Fore.GREEN}You: {Style.RESET_ALL}")
                
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                if not user_input.strip():
                    continue
                
                print(f"\n{Fore.CYAN}{model}: {Style.RESET_ALL}", end="", flush=True)
                
                # Get the response and print it character by character
                response = self.ollama_service.generate_response(model, user_input)
                
                # Print character by character
                for char in response:
                    print(char, end="", flush=True)
                    time.sleep(0.001)  # Small delay for visual effect
                
                print()  # Add a newline after the response
                
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
    
    def run(self):
        """Run the CLI application"""
        self.print_header()
        
        model = self.select_model()
        if model:
            self.chat_loop(model)
        
        self.console.print("\n[bold green]Thank you for using Oh-Llama! Goodbye![/]")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Oh-Llama: A colorful CLI for Ollama models")
    parser.add_argument("--model", "-m", help="Specify model to use directly")
    args = parser.parse_args()
    
    cli = OhLlamaCLI()
    
    if args.model:
        # Check if model exists or needs to be pulled
        local_models = OllamaService.get_local_models()
        model_exists = any(m["name"] == args.model for m in local_models)
        
        if not model_exists:
            console.print(f"[yellow]Model {args.model} not found locally. Pulling...[/]")
            if not OllamaService.pull_model(args.model):
                console.print(f"[red]Failed to pull model {args.model}[/]")
                return
        
        cli.chat_loop(args.model)
    else:
        cli.run()


if __name__ == "__main__":
    main()
