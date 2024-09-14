import os
import ast
import shutil
import re
import shlex
import sys
import discord
import glob
from discord import app_commands
import json
import aiohttp
import subprocess
import asyncio
from dotenv import load_dotenv, set_key
from collections import deque
from PIL import Image
from PIL.ExifTags import TAGS
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import ast

def cleanup_old_logs(log_folder, max_logs=10):
    # Get all log files
    log_files = glob.glob(os.path.join(log_folder, 'bot_*.log'))
    
    # Sort log files by modification time (oldest first)
    log_files.sort(key=os.path.getmtime)
    
    # Remove oldest files if we have more than max_logs
    while len(log_files) > max_logs:
        oldest_file = log_files.pop(0)
        os.remove(oldest_file)
        print(f"Deleted old log file: {oldest_file}")

# Set up logging
log_folder = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_folder, exist_ok=True)

# Create a timestamp for the log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_folder, f'bot_{timestamp}.log')
latest_log = os.path.join(log_folder, 'latest.log')

# Set up the logger
logger = logging.getLogger('discord_bot')
logger.setLevel(logging.INFO)

# Create a new file handler for each run
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

# Create or update the symlink to the latest log file
if os.path.exists(latest_log):
    os.remove(latest_log)
os.symlink(log_file, latest_log)

# Cleanup old log files, keeping only the 10 most recent
cleanup_old_logs(log_folder, max_logs=10)

# Log the start of a new session
logger.info(f"New session started. Logging to {log_file}")

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

class MyClient(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.command_queue = deque()
        self.queue_lock = asyncio.Lock()
        self.new_command_event = asyncio.Event()
        self.total_queue_length = 0
        self.current_job_progress = 0
        self.current_job_number = 0  

    async def setup_hook(self):
        await self.tree.sync()

    async def update_status(self):
        if self.total_queue_length > 0:
            status = f"Job {self.current_job_number} ({self.current_job_progress}%) of {self.total_queue_length}"
            await self.change_presence(activity=discord.Game(name=status))
        else:
            await self.change_presence(activity=None)

client = MyClient()

def get_allowed_users():
    allowed_users_str = os.getenv('ALLOWED_USERS', '')
    return [int(uid.strip()) for uid in allowed_users_str.split(',') if uid.strip()]

@client.event
async def on_ready():
    logger.info(f'{client.user} has connected to Discord!')
    logger.info(f'Connected to {len(client.guilds)} guilds')
    await client.tree.sync()
    logger.info('Command tree synced')
    client.loop.create_task(process_command_queue())
    logger.info('Command queue processing task started')

@client.tree.command()
async def delete_bot_messages(interaction: discord.Interaction, amount: str):
    ...
    """Delete a specified number of bot messages or all bot messages"""
    channel = interaction.channel
    logger.info(f'User {interaction.user.name} (ID: {interaction.user.id}) initiated delete_bot_messages command with amount: {amount}')
    await interaction.response.defer(ephemeral=True)

    try:
        if amount.lower() == 'all':
            delete_all = True
            limit = None
        else:
            delete_all = False
            limit = int(amount)
            if limit <= 0:
                raise ValueError("Amount must be a positive integer")
    except ValueError:
        logger.error('Invalid input for delete_bot_messages command')
        await interaction.followup.send("Invalid input. Please provide a positive integer or 'all'.", ephemeral=True)
        return

    deleted_count = 0
    async for message in interaction.channel.history(limit=None):
        if message.author == client.user:
            try:
                await message.delete()
                deleted_count += 1
                if not delete_all and deleted_count >= limit:
                    break
                await asyncio.sleep(1) 
            except discord.errors.HTTPException as e:
                if e.status == 429:  # Rate limit error
                    retry_after = e.retry_after
                    logger.warning(f"Rate limited, waiting for {retry_after:.2f} seconds")
                    await interaction.followup.send(f"Rate limited. Waiting for {retry_after:.2f} seconds before continuing.", ephemeral=True)
                    await asyncio.sleep(retry_after)
                else:
                    logger.error(f"Error occurred while deleting messages: {str(e)}")
                    await interaction.followup.send(f"An error occurred while deleting messages: {str(e)}", ephemeral=True)
                    break

    if delete_all:
        await interaction.followup.send(f"Deleted all {deleted_count} bot messages.", ephemeral=True)
    elif deleted_count < limit:
        await interaction.followup.send(f"Deleted {deleted_count} bot messages. There were fewer messages than requested.", ephemeral=True)
    else:
        await interaction.followup.send(f"Deleted {deleted_count} bot messages.", ephemeral=True)

    logger.info(f'Deleted {deleted_count} bot messages for user {interaction.user.name} (ID: {interaction.user.id})')
    
@client.tree.command()
async def clear_old_images(interaction: discord.Interaction):
    logger.info(f'User {interaction.user.name} (ID: {interaction.user.id}) initiated clear_old_images command')
    """Delete all files in the old image folder (Authorized users only)"""
    allowed_users = get_allowed_users()
    if interaction.user.id not in allowed_users:
        logger.warning(f'{interaction.user} tried to access clear_old_images but is not authorized.')
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    output_path_old = os.getenv('OUTPUT_PATH_OLD', 'old')
    
    if not os.path.exists(output_path_old):
        logger.info(f"The folder {output_path_old} does not exist.")
        await interaction.response.send_message(f"The folder {output_path_old} does not exist.", ephemeral=True)
        return

    files = os.listdir(output_path_old)
    file_count = len(files)

    if file_count == 0:
        logger.info(f"The folder {output_path_old} is already empty.")
        await interaction.response.send_message(f"The folder {output_path_old} is already empty.", ephemeral=True)
        return

    await interaction.response.send_message(
        f"Are you sure you want to delete all {file_count} files in the {output_path_old} folder? "
        "This action cannot be undone. Reply with 'yes' to confirm.",
        ephemeral=True
    )

    def check(m):
        return m.author.id == interaction.user.id and m.channel.id == interaction.channel.id

    try:
        msg = await client.wait_for('message', timeout=30.0, check=check)
    except asyncio.TimeoutError:
        logger.warning("Confirmation timed out. No files were deleted.")
        await interaction.followup.send("Confirmation timed out. No files were deleted.", ephemeral=True)
        return

    if msg.content.lower() != 'yes':
        logger.info("Operation cancelled. No files were deleted.")
        await interaction.followup.send("Operation cancelled. No files were deleted.", ephemeral=True)
        return

    deleted_count = 0
    for filename in files:
        file_path = os.path.join(output_path_old, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                deleted_count += 1
        except Exception as e:
            logger.error(f"Error deleting {filename}: {str(e)}")
            await interaction.followup.send(f"Error deleting {filename}: {str(e)}", ephemeral=True)

    logger.info(f"Successfully deleted {deleted_count} out of {file_count} files.")
    logger.info(f'User {interaction.user.name} (ID: {interaction.user.id}) confirmed deletion of {deleted_count} files')
    await interaction.followup.send(f"Successfully deleted {deleted_count} out of {file_count} files.", ephemeral=True)

@client.tree.command()
async def helpme(interaction: discord.Interaction):
    """Display help information for the Image Generation Bot"""
    help_text = """
**Image Generation Bot Help**

1. `/generate` command:
   Options:
   • model: Choose the model to use (default: schnell)
   • steps: Number of generation steps (default: 20 for dev, 5 for schnell)
   • seed: Random seed for generation (optional)
   • height: Image height in pixels (default: 1024 for dev, 512 for schnell)
   • width: Image width in pixels (default: 1024 for dev, 512 for schnell)
   • guidance: How closely it adheres to the prompt (default: 7 for dev, 1to20)
   • lora: Name of the LoRA file (optional)
   • prompt: Your image generation prompt (required)
   • inspiration: Number of AI-generated inspiration prompts (optional)

For more information or assistance, please contact the bot administrator.
    """
    logger.info(f"{interaction.user} requested help.")
    await interaction.response.send_message(help_text, ephemeral=True)

@client.tree.command()
@app_commands.describe(
    action="Choose 'view' to see current settings or 'edit' to modify a setting",
    setting="The setting to edit (only needed for 'edit' action)",
    value="The new value for the setting (only needed for 'edit' action)"
)
@app_commands.choices(action=[
    app_commands.Choice(name="View", value="view"),
    app_commands.Choice(name="Edit", value="edit")
])
@app_commands.choices(setting=[
    app_commands.Choice(name="Ollama Model", value="OLLAMA_MODEL"),
    app_commands.Choice(name="Ollama Temperature", value="OLLAMA_TEMPERATURE"),
    app_commands.Choice(name="Ollama Prompt Template", value="OLLAMA_PROMPT_TEMPLATE"),
    app_commands.Choice(name="Default Model", value="DEFAULT_MODEL"),
    app_commands.Choice(name="Dev Default Steps", value="DEV_DEFAULT_STEPS"),
    app_commands.Choice(name="Dev Default Height", value="DEV_DEFAULT_HEIGHT"),
    app_commands.Choice(name="Dev Default Width", value="DEV_DEFAULT_WIDTH"),
    app_commands.Choice(name="Default Guidance", value="DEFAULT_GUIDANCE"),
    app_commands.Choice(name="Schnell Default Steps", value="SCHNELL_DEFAULT_STEPS"),
    app_commands.Choice(name="Schnell Default Height", value="SCHNELL_DEFAULT_HEIGHT"),
    app_commands.Choice(name="Schnell Default Width", value="SCHNELL_DEFAULT_WIDTH"),
])
async def env(interaction: discord.Interaction, action: str, setting: str = None, value: str = None):
    """View or edit environment variables (Authorized users only)"""
    logger.info(f'User {interaction.user.name} (ID: {interaction.user.id}) initiated env command with action: {action}, setting: {setting}, value: {value}')

    allowed_users = get_allowed_users()
    if interaction.user.id not in allowed_users:
        logger.warning(f'{interaction.user} tried to access env but is not authorized.')
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    if action == 'view':
        env_vars = {
            "OLLAMA_MODEL": os.getenv('OLLAMA_MODEL'),
            "OLLAMA_TEMPERATURE": os.getenv('OLLAMA_TEMPERATURE'),
            "OLLAMA_PROMPT_TEMPLATE": os.getenv('OLLAMA_PROMPT_TEMPLATE'),
            "DEFAULT_MODEL": os.getenv('DEFAULT_MODEL'),
            "DEV_DEFAULT_STEPS": os.getenv('DEV_DEFAULT_STEPS'),
            "DEV_DEFAULT_HEIGHT": os.getenv('DEV_DEFAULT_HEIGHT'),
            "DEV_DEFAULT_WIDTH": os.getenv('DEV_DEFAULT_WIDTH'),
            "DEFAULT_GUIDANCE": os.getenv('DEFAULT_GUIDANCE'),
            "SCHNELL_DEFAULT_STEPS": os.getenv('SCHNELL_DEFAULT_STEPS'),
            "SCHNELL_DEFAULT_HEIGHT": os.getenv('SCHNELL_DEFAULT_HEIGHT'),
            "SCHNELL_DEFAULT_WIDTH": os.getenv('SCHNELL_DEFAULT_WIDTH'),
        }
        message = "Current environment variables:\n```\n"
        for k, v in env_vars.items():
            message += f"{k}: {v}\n"
        message += "```"
        logger.info(f"{interaction.user} viewed environment variables.")
        await interaction.response.send_message(message, ephemeral=True)

    elif action == 'edit':
        if not setting or not value:
            await interaction.response.send_message("Both 'setting' and 'value' are required for editing.", ephemeral=True)
            return

        # Update the environment variable
        os.environ[setting] = value

        # Update the .env file
        env_path = '/Users/themastermind/mflux/scripts/.env'
        updated = False
        if os.path.exists(env_path):
            with open(env_path, 'r') as file:
                lines = file.readlines()
            with open(env_path, 'w') as file:
                for line in lines:
                    if line.startswith(f"{setting}="):
                        file.write(f"{setting}={value}\n")
                        updated = True
                    else:
                        file.write(line)
                if not updated:
                    file.write(f"{setting}={value}\n")
        else:
            with open(env_path, 'w') as file:
                file.write(f"{setting}={value}\n")

        logger.info(f"Environment variable '{setting}' has been updated to '{value}' by {interaction.user}.")
        await interaction.response.send_message(f"Environment variable '{setting}' has been updated to '{value}'.", ephemeral=True)

@client.tree.command()
async def reload(interaction: discord.Interaction):
    """Reload the bot configuration (Authorized users only)"""
    allowed_users = get_allowed_users()
    if interaction.user.id not in allowed_users:
        logger.warning(f'{interaction.user} tried to reload configuration but is not authorized.')
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return
    
    reload_config()
    logger.info(f"{interaction.user} reloaded the bot configuration.")
    await interaction.response.send_message("Configuration reloaded successfully!", ephemeral=True)

def reload_config():
    # Reload environment variables
    load_dotenv(override=True)
    
    # Update any other configuration settings that depend on environment variables
    # For example:
    global SOME_GLOBAL_CONFIG
    SOME_GLOBAL_CONFIG = os.getenv('SOME_ENV_VAR', 'default_value')
    
    logger.info("Configuration reloaded.")

@client.tree.command()
@app_commands.describe(
    model="Choose the model to use (dev or schnell)",
    steps="Number of generation steps",
    seed="Random seed for generation (optional)",
    height="Image height in pixels",
    width="Image width in pixels",
    guidance="1-20, how closely it adheres to the prompt (optional)",
    lora="Name of the LoRA file",
    prompt="Your image generation prompt",
    inspiration="Number of AI-generated inspiration prompts"
)
async def generate(interaction: discord.Interaction, model: str = "schnell", steps: int = None, seed: int = None, 
                   height: int = None, width: int = None, guidance: int = None, lora: str = None, prompt: str = None, inspiration: int = 0):
    channel = interaction.channel
    logger.info(f'User {interaction.user.name} (ID: {interaction.user.id}) initiated generate command')
    logger.info(f'Generate parameters: model={model}, steps={steps}, seed={seed}, height={height}, width={width}, guidance={guidance}, lora={lora}, inspiration={inspiration}')
    logger.info(f'Prompt: {prompt}')
    await interaction.response.defer(thinking=True)

    # Set default values based on the model
    if model == "dev":
        steps = steps or int(os.getenv('DEV_DEFAULT_STEPS'))
        height = height or int(os.getenv('DEV_DEFAULT_HEIGHT'))
        width = width or int(os.getenv('DEV_DEFAULT_WIDTH'))
        guidance = guidance or int(os.getenv('DEFAULT_GUIDANCE'))
    else:  # schnell
        steps = steps or int(os.getenv('SCHNELL_DEFAULT_STEPS'))
        height = height or int(os.getenv('SCHNELL_DEFAULT_HEIGHT'))
        width = width or int(os.getenv('SCHNELL_DEFAULT_WIDTH'))

    user_input = f"User Input:\nModel: {model}, Steps: {steps}, Seed: {seed}, Height: {height}, Width: {width}, guidance={guidance}, LoRA: {lora}, Inspiration: {inspiration}\nPrompt: {prompt}\n\n"

    prompts = []
    if inspiration > 0:
        for _ in range(inspiration):
            generated_prompt = await generate_prompt(os.getenv('OLLAMA_MODEL'), prompt)
            prompts.append(generated_prompt)
    else:
        prompts = [prompt]

    async with client.queue_lock:
        base_position = client.total_queue_length + (1 if client.current_job_number == 0 else client.current_job_number)
        new_commands = len(prompts)
        client.total_queue_length += new_commands
        username = interaction.user.name
        for i, p in enumerate(prompts):
            position = base_position + i
            client.command_queue.append((interaction, model, steps, seed, height, width, guidance, lora, p, position, user_input, username))

    await client.update_status()

    if inspiration > 0:
        logger.info(f"{interaction.user} initiated generation with AI-inspired prompts.")
        await interaction.followup.send(f"{user_input}{inspiration} AI-inspired prompts have been added to the queue based on your input.", ephemeral=True)
    else:
        logger.info(f"{interaction.user} initiated generation.")
        await interaction.followup.send(f"{user_input}Your command has been added to the queue.", ephemeral=True)
    
    client.new_command_event.set()

async def generate_prompt(model, base_prompt):
    prompt = os.getenv('OLLAMA_PROMPT_TEMPLATE')
    prompt = prompt.format(base_prompt=base_prompt)
    temperature = float(os.getenv('OLLAMA_TEMPERATURE', 0.7))  # Default to 0.7 if not set

    logger.info(f'Generating prompt using model: {model}')
    logger.info(f'Base prompt: {base_prompt}')
    logger.info(f'Formatted prompt: {prompt}')
    logger.info(f'Temperature: {temperature}')

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{os.getenv('OLLAMA_API_URL')}/api/generate", json={
            'model': model,
            'prompt': prompt,
            'stream': False,
            'temperature': temperature
        }) as response:
            if response.status != 200:
                logger.error(f"Error generating prompt: {await response.text()}")
                return base_prompt

            try:
                data = await response.json()
                generated_prompt = data['response'].strip()
                logger.info(f'Generated prompt: {generated_prompt}')
                return generated_prompt
            except Exception as e:
                logger.error(f"Error processing response: {str(e)}")
                return base_prompt

async def process_command_queue():
    while True:
        if not client.command_queue:
            logger.info('Command queue is empty, waiting for new commands')
            client.current_job_number = 0
            client.current_job_progress = 0
            client.total_queue_length = 0  # Reset total queue length when empty
            await client.update_status()
            await client.new_command_event.wait()
            client.new_command_event.clear()

        async with client.queue_lock:
            if client.command_queue:
                logger.info(f'Processing next command in queue. Queue length: {len(client.command_queue)}')
                interaction, model, steps, seed, height, width, guidance, lora, prompt, position, user_input, username = client.command_queue.popleft()
                client.current_job_number = position
            else:
                continue
        
        await generate_image(interaction, model, steps, seed, height, width, guidance, lora, prompt, position, user_input, username)
        
        # Remove this line:
        # client.total_queue_length -= 1
        
        await client.update_status()

def get_exif_data(image_path):
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data:
                labeled = {TAGS.get(k, k): v for k, v in exif_data.items()}
                user_comment = labeled.get('UserComment')
                if user_comment:
                    if isinstance(user_comment, bytes):
                        try:
                            user_comment = user_comment.decode('utf-8')
                        except UnicodeDecodeError:
                            user_comment = user_comment.decode('utf-8', 'ignore')
                    labeled['UserComment'] = user_comment
                return labeled
    except Exception as e:
        logger.error(f"Error reading EXIF data: {str(e)}")
    return {}

def parse_user_comment(user_comment):
    if not user_comment:
        return {}

    # Try parsing as JSON
    try:
        return json.loads(user_comment)
    except json.JSONDecodeError:
        pass

    # Try parsing as Python literal
    try:
        return ast.literal_eval(user_comment)
    except (ValueError, SyntaxError):
        pass

    # Try parsing as key-value pairs
    if isinstance(user_comment, str):
        try:
            return dict(item.split(": ") for item in user_comment.split(", "))
        except ValueError:
            pass

    # If all else fails, return the raw string
    return {"raw_comment": user_comment}

async def generate_image(interaction, model, steps, seed, height, width, guidance, lora, prompt, position, user_input, username):
    logger.info(f"Starting image generation for {username}")
    
    channel = interaction.channel
    
    cmd = [
        "mflux-generate",
        "--path", os.getenv(f'{model.upper()}_PATH'),
        "--model", model,
        "--output", os.getenv('OUTPUT_PATH', 'image.png'),
        "--steps", str(steps),
        "--height", str(height),
        "--width", str(width),
        "--prompt", prompt
    ]

    if model == "dev":
        cmd.extend(["--guidance", str(guidance),])

    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if lora:
        cmd.extend(["--lora-paths", f"{os.getenv('LORA_PATH')}/{lora}.safetensors"])

    output_path = os.getenv('OUTPUT_PATH', 'image.png')
    if os.path.exists(output_path):
        logger.info(f"Removed leftover image")
        save_image_copy(output_path, username)
        os.remove(output_path)

    cmd_str = " ".join(shlex.quote(str(arg)) for arg in cmd)
    logger.info(f"Executing command: {cmd_str}")

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        async def read_stream(stream):
            buffer = ""
            while True:
                chunk = await stream.read(1024)
                if not chunk:
                    break
                buffer += chunk.decode('utf-8')
                lines = buffer.split('\r')
                buffer = lines.pop()

                for line in lines:
                    match = re.search(r'(\d+)%\|', line)
                    if match:
                        percentage = int(match.group(1))
                        client.current_job_progress = percentage
                        await client.update_status()

        await asyncio.gather(
            read_stream(process.stdout),
            read_stream(process.stderr)
        )

        await process.wait()

        if process.returncode != 0:
            stdout = await process.stdout.read()
            stderr = await process.stderr.read()
            error_message = f"Error occurred:\nStdout: {stdout.decode()}\nStderr: {stderr.decode()}"
            logger.error(error_message)
            await interaction.followup.send(error_message)
            return

        if os.path.exists(output_path):
            exif_data = get_exif_data(output_path)
            comment_data = parse_user_comment(exif_data.get('UserComment', ''))

            formatted_data = format_image_details(comment_data)
            success_message = f"{user_input}Image generated successfully!\n\n```{formatted_data}```"

            save_image_copy(output_path, username)

            await channel.send(content=success_message, file=discord.File(output_path))
            os.remove(output_path)
            logger.info(f"Image generation completed for {username}")
        else:
            raise FileNotFoundError("Generated image file not found")

    except Exception as e:
        error_message = f"{user_input}An error occurred during image generation: {str(e)}"
        logger.error(error_message)
        await channel.send(error_message)

    finally:
        client.current_job_progress = 0
        await client.update_status()

def get_exif_data(image_path):
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data:
                return {TAGS.get(k, k): v for k, v in exif_data.items()}
    except Exception as e:
        logger.error(f"Error reading EXIF data: {str(e)}")
    return {}
    
def parse_user_comment(user_comment):
    if not user_comment:
        return {}

    # If it's bytes, decode to string
    if isinstance(user_comment, bytes):
        try:
            user_comment = user_comment.decode('utf-8')
        except UnicodeDecodeError:
            user_comment = user_comment.decode('utf-8', 'ignore')

    # Try parsing as JSON
    try:
        return json.loads(user_comment)
    except json.JSONDecodeError:
        pass

    # Try parsing as Python literal
    try:
        return ast.literal_eval(user_comment)
    except (ValueError, SyntaxError):
        pass

    # Try parsing as key-value pairs
    if isinstance(user_comment, str):
        try:
            return dict(item.split(": ") for item in user_comment.split(", "))
        except ValueError:
            pass

    # If all else fails, return the raw string
    return {"raw_comment": user_comment}

def format_image_details(comment_data):
    formatted_data = "Image Generation Details:\n"
    for key in ['model', 'seed', 'steps', 'guidance', 'generation_time', 'lora_paths', 'lora_scales', 'prompt']:
        value = comment_data.get(key, 'N/A')
        formatted_data += f"• {key.capitalize()}: {value}\n"
    return formatted_data

def save_image_copy(output_path, username):
    output_path_old = os.getenv('OUTPUT_PATH_OLD', 'old')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_username = ''.join(c for c in username if c.isalnum() or c in ('-', '_'))
    new_filename = f"{safe_username}_{timestamp}.png"
    new_path = os.path.join(output_path_old, new_filename)
    os.makedirs(output_path_old, exist_ok=True)
    shutil.copy2(output_path, new_path)
    logger.info(f"Copied generated image to: {new_path}")

try:
    client.run(os.getenv('DISCORD_TOKEN'))
except Exception as e:
    logger.critical(f'Failed to start the bot: {str(e)}')
    sys.exit(1)