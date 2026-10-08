import os
import re
import spacy
import lemminflect
import asyncio
import logging
import discord
import random

from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
from services.llm import ask_llm, translate_to_chinese
from pint import UnitRegistry


# -----------------------------
# Environment
# -----------------------------

nlp = spacy.load("en_core_web_sm")
ureg = UnitRegistry()

load_dotenv(override=True)

TOKEN = os.getenv("DISCORD_TOKEN")

if not TOKEN:
    raise RuntimeError("DISCORD_TOKEN is not set")

# -----------------------------
# Logging
# -----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logger = logging.getLogger("heavenly-principles")


# -----------------------------
# Global configuration
# -----------------------------

# Maximum size of a question sent to the LLM
MAX_QUESTION_LENGTH = 2000

# Maximum size of a replied-to message that utility commands will process
MAX_REPLY_LENGTH = 4000

# Maximum amount of time we'll wait for an LLM response
LLM_TIMEOUT_SECONDS = 45

# Maximum number of LLM calls running at the same time
MAX_CONCURRENT_LLM_REQUESTS = 3

llm_semaphore = asyncio.Semaphore(
    MAX_CONCURRENT_LLM_REQUESTS
)

TRANSLATE_CHANCE = 0.03  # 3% of messages
REACTION_CHANCE = 0.05  # 5% of messages

# -----------------------------
# Paimonify
# -----------------------------

def paimonify_text(text: str) -> str:
    doc = nlp(text)
    replacements = {}

    for token in doc:

        # Replace first-person pronouns
        if token.lower_ == "i":
            replacements[token.i] = "Paimon"

            if token.dep_ in ("nsubj", "nsubjpass"):
                verb = token.head

                # Check whether this verb is governed by a modal
                has_modal = any(
                    child.dep_ == "aux"
                    and child.tag_ == "MD"
                    for child in verb.children
                )

                # Only change simple present-tense verbs
                if verb.tag_ == "VBP" and not has_modal:
                    inflected = verb._.inflect("VBZ")

                    if inflected:
                        replacements[verb.i] = inflected

        elif token.lower_ == "me":
            replacements[token.i] = "Paimon"

        elif token.lower_ == "my":
            replacements[token.i] = "Paimon's"

        elif token.lower_ == "mine":
            replacements[token.i] = "Paimon's"

        elif token.lower_ == "myself":
            replacements[token.i] = "Paimon"

        # Handle "I am" -> "Paimon is"
        elif token.lower_ == "am":
            subject = next(
                (
                    child
                    for child in token.children
                    if child.dep_ in ("nsubj", "nsubjpass")
                    and child.lower_ == "i"
                ),
                None,
            )

            if subject:
                replacements[token.i] = "is"

    output = []

    for token in doc:
        replacement = replacements.get(
            token.i,
            token.text
        )

        output.append(
            replacement + token.whitespace_
        )

    transformed = "".join(output)

    return f"*{transformed}* (๑˃ᴗ˂)ﻭ"


# -----------------------------
# Imperial -> Metric
# -----------------------------

def convert_to_metric(text: str) -> str:
    conversions = {
        # Temperature
        "°f": "degC",
        "f": "degC",
        "fahrenheit": "degC",

        # Length
        "in": "cm",
        "inch": "cm",
        "inches": "cm",

        "ft": "m",
        "foot": "m",
        "feet": "m",

        "yd": "m",
        "yard": "m",
        "yards": "m",

        "mi": "km",
        "mile": "km",
        "miles": "km",

        # Weight / mass
        "oz": "g",
        "ounce": "g",
        "ounces": "g",

        "lb": "kg",
        "lbs": "kg",
        "pound": "kg",
        "pounds": "kg",

        # Volume
        "fl oz": "ml",
        "fluid ounce": "ml",
        "fluid ounces": "ml",

        "cup": "ml",
        "cups": "ml",

        "pt": "l",
        "pint": "l",
        "pints": "l",

        "qt": "l",
        "quart": "l",
        "quarts": "l",

        "gal": "l",
        "gallon": "l",
        "gallons": "l",
    }

    pattern = re.compile(
        r"(-?\d+(?:\.\d+)?)\s*"
        r"(°F|F|fahrenheit|"
        r"fluid ounces?|fl oz|"
        r"inches?|in|"
        r"feet|foot|ft|"
        r"yards?|yd|"
        r"miles?|mi|"
        r"ounces?|oz|"
        r"pounds?|lbs?|"
        r"cups?|"
        r"pints?|pt|"
        r"quarts?|qt|"
        r"gallons?|gal)\b",
        re.IGNORECASE,
    )

    def replace(match):
        value = float(match.group(1))
        original_unit = match.group(2)
        unit = original_unit.lower()

        target_unit = conversions.get(unit)

        if not target_unit:
            return match.group(0)

        try:

            # Fahrenheit needs special handling
            if unit in {
                "°f",
                "f",
                "fahrenheit"
            }:
                metric_value = (
                    (value - 32) * 5 / 9
                )

                return f"{metric_value:.1f} °C"

            quantity = value * ureg(unit)

            converted = quantity.to(
                target_unit
            )

            metric_value = converted.magnitude

            # Keep output readable
            if abs(metric_value) >= 100:
                formatted = (
                    f"{metric_value:.0f}"
                )

            elif abs(metric_value) >= 10:
                formatted = (
                    f"{metric_value:.1f}"
                )

            else:
                formatted = (
                    f"{metric_value:.2f}"
                )

            display_units = {
                "cm": "cm",
                "m": "m",
                "km": "km",
                "g": "g",
                "kg": "kg",
                "ml": "mL",
                "l": "L",
            }

            return (
                f"{formatted} "
                f"{display_units[target_unit]}"
            )

        except Exception:
            logger.exception(
                "Metric conversion failed"
            )

            return match.group(0)

    return pattern.sub(
        replace,
        text
    )

# -----------------------------
# uwuify
# -----------------------------

def uwuify_text(text: str) -> str:
    # Replace non-leading r/l with w inside each word
    def transform_word(match):
        word = match.group(0)

        if len(word) <= 1:
            return word

        body = word[:-1]
        last = word[-1]

        body = re.sub(r"[rl]", "w", body)
        body = re.sub(r"[RL]", "W", body)

        return body + last

    transformed = re.sub(
        r"\b[A-Za-z]+\b",
        transform_word,
        text
    )

    # Add "uwu" to the end of each sentence
    transformed = re.sub(
        r"([.!?]+)(?=\s|$)",
        r" uwu\1",
        transformed
    )

    # If the text doesn't end in sentence punctuation,
    # still append lol
    if not re.search(r"[.!?]\s*$", transformed):
        transformed = transformed.rstrip() + " uwu~"

    return f"*{transformed}*"

# -----------------------------
# Discord bot setup
# -----------------------------

intents = discord.Intents.default()

# Required for !prefix commands
intents.message_content = True

bot = commands.Bot(
    command_prefix="!",
    intents=intents,
)


# -----------------------------
# Helpers
# -----------------------------

async def get_replied_message(
    ctx: commands.Context
):
    """
    Returns the message that the command
    was replying to.
    """

    if ctx.message.reference is None:
        return None

    referenced_message = (
        ctx.message.reference.resolved
    )

    if referenced_message is None:
        try:
            referenced_message = (
                await ctx.channel.fetch_message(
                    ctx.message.reference.message_id
                )
            )

        except discord.NotFound:
            return None

        except discord.Forbidden:
            return None

        except discord.HTTPException:
            logger.exception(
                "Failed to fetch replied-to message"
            )
            return None

    return referenced_message


async def send_long_response(
    interaction: discord.Interaction,
    text: str
):
    """
    Discord has a 2000-character message limit.
    Split long LLM responses into chunks.
    """

    if not text:
        return

    for i in range(0, len(text), 2000):
        await interaction.followup.send(
            text[i:i + 2000]
        )


# -----------------------------
# Global prefix-command protection
# -----------------------------

@bot.check
async def ignore_bot_users(
    ctx: commands.Context
):
    """
    Stops other bots from triggering
    prefix commands.
    """

    return not ctx.author.bot


# -----------------------------
# Startup
# -----------------------------

@bot.event
async def on_ready():
    logger.info(
        "Logged in as %s",
        bot.user
    )

    try:
        synced = await bot.tree.sync()

        logger.info(
            "Synced %s slash commands",
            len(synced)
        )

    except Exception:
        logger.exception(
            "Failed to sync slash commands"
        )


# -----------------------------
# /paimon_ping
# -----------------------------

@bot.tree.command(
    name="paimon_ping",
    description="Check if the Heavenly Principles are awake",
)
@app_commands.checks.cooldown(
    5,
    60.0,
    key=lambda interaction: interaction.user.id
)
async def paimon_ping(
    interaction: discord.Interaction
):
    await interaction.response.send_message(
        "pong"
    )


# -----------------------------
# /paimon_ask
# -----------------------------

@bot.tree.command(
    name="paimon_ask",
    description="Ask the Heavenly Principles a question",
)
@app_commands.checks.cooldown(
    3,
    60.0,
    key=lambda interaction: interaction.user.id
)
async def paimon_ask(
    interaction: discord.Interaction,
    question: str,
):

    # -------------------------
    # Input validation
    # -------------------------

    question = question.strip()

    if not question:
        await interaction.response.send_message(
            "Please enter a question.",
            ephemeral=True
        )
        return

    if len(question) > MAX_QUESTION_LENGTH:
        await interaction.response.send_message(
            (
                "That question is too long. "
                f"Please keep it under "
                f"{MAX_QUESTION_LENGTH} characters."
            ),
            ephemeral=True
        )
        return

    # Tell Discord we're processing
    await interaction.response.defer()

    try:

        # Hard timeout
        async with asyncio.timeout(
            LLM_TIMEOUT_SECONDS
        ):

            # Global concurrency protection
            async with llm_semaphore:

                logger.info(
                    "%s used /paimon_ask in guild %s",
                    interaction.user,
                    (
                        interaction.guild.id
                        if interaction.guild
                        else "DM"
                    )
                )

                answer = await ask_llm(
                    question
                )

        if not answer:
            await interaction.followup.send(
                "I couldn't generate a response."
            )
            return

        formatted_response = (
            f"**Question:** {question}\n"
            f"**Answer:**\n{answer}"
        )

        await send_long_response(
            interaction,
            formatted_response
        )

    except TimeoutError:
        logger.warning(
            "LLM request timed out for user %s",
            interaction.user
        )

        await interaction.followup.send(
            "That request took too long. Try again."
        )

    except Exception:
        logger.exception(
            "/paimon_ask failed"
        )

        await interaction.followup.send(
            "Something went wrong while generating the response."
        )


# -----------------------------
# !paimonify
# -----------------------------

@bot.command(
    name="paimonify"
)
@commands.cooldown(
    5,
    60,
    commands.BucketType.user
)
async def paimonify(
    ctx: commands.Context
):

    referenced_message = (
        await get_replied_message(ctx)
    )

    if referenced_message is None:
        await ctx.reply(
            "Reply to a message first, "
            "then use `!paimonify`."
        )
        return

    original = referenced_message.content

    if not original:
        await ctx.reply(
            "That message has no text to process."
        )
        return

    if len(original) > MAX_REPLY_LENGTH:
        await ctx.reply(
            "That message is too long to process."
        )
        return

    try:
        transformed = paimonify_text(
            original
        )

        await referenced_message.reply(
            transformed
        )

        logger.info(
            "%s used !paimonify in guild %s",
            ctx.author,
            (
                ctx.guild.id
                if ctx.guild
                else "DM"
            )
        )

    except Exception:
        logger.exception(
            "!paimonify failed"
        )

        await ctx.reply(
            "Something went wrong while processing that message."
        )


# -----------------------------
# !untard
# -----------------------------

@bot.command(
    name="unretard"
)
@commands.cooldown(
    5,
    60,
    commands.BucketType.user
)
async def metric(
    ctx: commands.Context
):

    referenced_message = (
        await get_replied_message(ctx)
    )

    if referenced_message is None:
        await ctx.reply(
            "Reply to a message containing "
            "an imperial measurement, "
            "then use `!unretard`."
        )
        return

    original = referenced_message.content

    if not original:
        await ctx.reply(
            "That message has no text to process."
        )
        return

    if len(original) > MAX_REPLY_LENGTH:
        await ctx.reply(
            "That message is too long to process."
        )
        return

    try:
        converted = convert_to_metric(
            original
        )

        if converted == original:
            await ctx.reply(
                "I couldn't find any imperial "
                "measurements to convert."
            )
            return

        await referenced_message.reply(
            converted
        )

        logger.info(
            "%s used !untard in guild %s",
            ctx.author,
            (
                ctx.guild.id
                if ctx.guild
                else "DM"
            )
        )

    except Exception:
        logger.exception(
            "!untard failed"
        )

        await ctx.reply(
            "Something went wrong while "
            "converting that message."
        )

# -----------------------------
# !uwuify
# -----------------------------

@bot.command(name="uwuify")
@commands.cooldown(
    5,
    60,
    commands.BucketType.user
)
async def uwu(ctx: commands.Context):

    referenced_message = (
        await get_replied_message(ctx)
    )

    if referenced_message is None:
        await ctx.reply(
            "Reply to a message first, then use `!uwuify`."
        )
        return

    original = referenced_message.content

    if not original:
        await ctx.reply(
            "That message has no text to process."
        )
        return

    if len(original) > MAX_REPLY_LENGTH:
        await ctx.reply(
            "That message is too long to process."
        )
        return

    try:
        transformed = uwuify_text(original)

        await referenced_message.reply(
            transformed
        )

        logger.info(
            "%s used !uwu in guild %s",
            ctx.author,
            ctx.guild.id if ctx.guild else "DM"
        )

    except Exception:
        logger.exception(
            "!uwu failed"
        )

        await ctx.reply(
            "Something went wrong while processing that message."
        )

# -----------------------------
# !duel
# -----------------------------

@bot.command(name="duelpair")
@commands.cooldown(
    5,
    60,
    commands.BucketType.user
)
async def duel(
    ctx: commands.Context,
    *,
    matchup: str
):
    if "/" not in matchup:
        await ctx.reply(
            "Use `!duelpair option A/option B`."
        )
        return

    option_a, option_b = matchup.split("/", 1)

    option_a = option_a.strip()
    option_b = option_b.strip()

    if not option_a or not option_b:
        await ctx.reply(
            "Both sides of the duel need a value."
        )
        return

    winner = random.choice(
        [option_a, option_b]
    )

    if winner == option_a:
        loser = option_b
    else:
        loser = option_a


    messages = [
        f"**{winner} got the Victory Royale.**",
        f"**{winner} got sent to Ram Ranch.**",
        f"**{winner} is going to Brazil.**",
        f"**{winner} did not win the 50/50.**",
        f"**{winner} received a free trip to a Private Island™.**",
        f"**{winner} received a mandatory invitation to a Diddy Party.**",
        f"**{winner} won the 50/50 in the greatest gacha ever.**",
        f"**Sending Scihub to {winner}'s location.**",
        f"**{winner} was forced to wait for Azur Promelia global release.**",
        f"**{winner} was turned into a marketable plushie.**",
        f"**{winner} became a Zenless Zone Zero stunner.**",
        f"**{winner} was forced to update Prydwen for Honkai Star Rail.**",
        f"**{winner} NTR'd all of {loser}'s waifus in Mudae.**",
        f"**{winner} was forced to consume slop.**",
        f"**{winner} won a billion primogems (for real this time).**",
        f"**Internet artists genderbent {winner}.**",
        f"**If you goon, {winner} dies. Of course, we gooned.**",
        f"**{winner} vanished searching for Nobu.**",
        f"**{winner} vanished searching for Sed.**",
        f"**{winner} vanished searching for Loli.**",
        f"**{winner}'s favorite live-service game announced EOS.**",
        f"**{winner} opened a Mr. Beast Discord Image.**",
        f"**{winner} got W Groomed at a Smash Ultimate tournament.**",
        f"**Get ready to learn Chinese, {winner}.**",
        f"**Mihoyo satellite targeting system online: {winner} will be terminated in 15 seconds.**",
        f"**{winner} gambled the house away.**",
        f"**{winner} in fact, could not handle allat.**",
        f"**Carl revealed {winner} is gay.**",
        f"**Gorlock sat on {loser}.**",
        f"**{winner} ejected {loser} from an airlock.**",
        f"**{winner} fired {loser} into the sun.**",
        f"**{winner} sent {loser} to #blue-archive.**",
        f"**{winner} powercrept {loser} into T5.**",
        f"**{winner} plapped {loser}.**",
        f"**{winner}'s Discord account was flagged for inappropriate content.**",
        f"**With this treasure, {winner} summoned and was beaten by Mahoraga.**",
        f"**{winner} was disassembled by Grace Howard!**",
        f"**{winner} got tag-teamed by Hu Tao, Sparkle, Burnice, and Yuzuha.**",
        f"**Carl revealed {winner} is not gay.**",
        f"**{winner} is the Lebron James of randomly selected options.**",
        f"**Steam Status: {winner} is now playing: Femboy Futa House.**",
        f"**{winner} was attacked by a kemonomimi in the woods. All of a sudden they have to take care of dozens of wolf children! Every night their number grows! Fail to do so and their partner, the Wolf Mama, will kill them! How many can they sustain?**"
    ]

    await ctx.reply(
        random.choice(messages)
    )

    logger.info(
        "%s used !duel: %s vs %s -> %s",
        ctx.author,
        option_a,
        option_b,
        winner
    )

# -----------------------------
# !8ball
# -----------------------------

@bot.command(name="8ball")
@commands.cooldown(
    5,
    30.0,
    commands.BucketType.user
)
async def eight_ball(
    ctx: commands.Context,
    *,
    question: str = None,
):
    if not question:
        await ctx.reply(
            "Ask Paimon a yes-or-no question first!"
        )
        return

    answer = random.choice([
        "Yes.",
        "No.",
    ])

    await ctx.reply(
        f"🎱 Paimon says: **{answer}**"
    )

# -----------------------------
# Im dad!
# -----------------------------


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    triggered = False

    # Dadbot feature
    dadmatch = re.search(
        r"\b(?:i['’]?m|i am)\s+(.+)",
        message.content,
        re.IGNORECASE
    )

    if dadmatch:
        x = dadmatch.group(1).strip()

        if x:
            await message.reply(
                f"Hi {x}, I'm Paimon!"
            )
            triggered = True

    # Random Chinese translation
    if (
        not triggered
        and message.content.strip()
        and not message.content.startswith("!")
        and random.random() <= TRANSLATE_CHANCE
    ):
        try:
            translation = await translate_to_chinese(
                message.content
            )

            await message.reply(
                f"{translation}"
            )

        except Exception:
            logger.exception(
                "Random Chinese translation failed."
            )
      # Random emoji reaction

    if (
        not triggered
        and message.content.strip()
        and not message.content.startswith("!")
        and random.random() <= REACTION_CHANCE
    ):
        try:
            emoji = "😂"
            await message.add_reaction(emoji)

        except Exception:
            logger.exception(
                "Random emoji reaction failed."
            )


    await bot.process_commands(message)

# -----------------------------
# Prefix command error handling
# -----------------------------

@bot.event
async def on_command_error(
    ctx: commands.Context,
    error
):

    # User exceeded cooldown
    if isinstance(
        error,
        commands.CommandOnCooldown
    ):
        await ctx.reply(
            (
                "Slow down — try again in "
                f"{error.retry_after:.1f}s."
            )
        )
        return

    # User lacks required Discord permissions
    if isinstance(
        error,
        commands.MissingPermissions
    ):
        await ctx.reply(
            "You don't have permission "
            "to use that command."
        )
        return

    # Ignore random !commands
    if isinstance(
        error,
        commands.CommandNotFound
    ):
        return

    logger.error(
        "Unhandled prefix command error",
        exc_info=(
            type(error),
            error,
            error.__traceback__
        )
    )

    try:
        await ctx.reply(
            "Something went wrong."
        )

    except discord.HTTPException:
        pass


# -----------------------------
# Slash command error handling
# -----------------------------

@bot.tree.error
async def on_app_command_error(
    interaction: discord.Interaction,
    error: app_commands.AppCommandError
):

    if isinstance(
        error,
        app_commands.CommandOnCooldown
    ):
        message = (
            "Slow down — try again in "
            f"{error.retry_after:.1f}s."
        )

    elif isinstance(
        error,
        app_commands.MissingPermissions
    ):
        message = (
            "You don't have permission "
            "to use that command."
        )

    else:
        logger.error(
            "Unhandled slash command error",
            exc_info=(
                type(error),
                error,
                error.__traceback__
            )
        )

        message = (
            "Something went wrong."
        )

    try:

        if interaction.response.is_done():

            await interaction.followup.send(
                message,
                ephemeral=True
            )

        else:

            await interaction.response.send_message(
                message,
                ephemeral=True
            )

    except discord.HTTPException:
        logger.exception(
            "Failed to send slash-command "
            "error response"
        )


# -----------------------------
# Start bot
# -----------------------------

bot.run(TOKEN)