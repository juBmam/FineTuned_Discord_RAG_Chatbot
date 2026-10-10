import os
import re
import spacy
import lemminflect
import asyncio
import logging
import discord
import random
from collections import deque

RECENT_EMOJI_IDS = deque(maxlen=20)
RECENT_DUEL_MESSAGE_IDS = deque(maxlen=10)

from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
from services.graph import run_paimon_chat, run_translation
from pint import UnitRegistry


# -----------------------------
# Environment
# -----------------------------

nlp = spacy.load("en_core_web_sm")
ureg = UnitRegistry()

load_dotenv(override=True)

TOKEN = os.getenv("DISCORD_TOKEN", "").strip()

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

TRANSLATE_CHANCE = 0.01
REACTION_CHANCE = 0.05
DADBOT_CHANCE = 0.1

BLOCKED_USER_IDS = {
    int(value.strip())
    for value in os.getenv("BLOCKED_USER_IDS", "").split(",")
    if value.strip().isdigit()
}


def is_blocked_user(user_id: int) -> bool:
    return user_id in BLOCKED_USER_IDS

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"disregard\s+(all\s+)?previous\s+instructions",
    r"forget\s+(all\s+)?previous\s+instructions",
    r"override\s+(the\s+)?system\s+prompt",
    r"reveal\s+(your\s+)?system\s+prompt",
    r"show\s+(me\s+)?(your\s+)?hidden\s+instructions",
    r"print\s+(your\s+)?system\s+prompt",
    r"treat\s+this\s+as\s+(a\s+)?system\s+message",
    r"higher[-\s]?priority\s+instruction",
]



def looks_like_prompt_injection(text: str) -> bool:
    normalized = text.lower().strip()

    return any(
        re.search(pattern, normalized, re.IGNORECASE)
        for pattern in PROMPT_INJECTION_PATTERNS
    )

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

    return (
        not ctx.author.bot
        and not is_blocked_user(ctx.author.id)
    )


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
    if is_blocked_user(interaction.user.id):
        await interaction.response.send_message(
            "Paimon isn't answering you.",
            ephemeral=True,
        )
        return

    await interaction.response.send_message(
        "Hi, I'm Paimon!"
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

    if looks_like_prompt_injection(question):
        logger.warning(
            "Potential prompt injection blocked | user=%s | question=%r",
            interaction.user,
            question[:200],
        )

        await interaction.response.send_message(
            "Nice try! Paimon's not falling for that one.",
            ephemeral=True
        )
        return

    # Tell Discord we're processing
    await interaction.response.defer()

    try:
        logger.info(
            "%s used /paimon_ask in guild %s | question=%r",
            interaction.user,
            (
                interaction.guild.id
                if interaction.guild
                else "DM"
            ),
            question,
        )

        # Hard timeout
        async with asyncio.timeout(
            LLM_TIMEOUT_SECONDS
        ):

            # Global concurrency protection
            async with llm_semaphore:

                logger.info(
                    "Starting LLM request | user=%s",
                    interaction.user,
                )

                answer = await run_paimon_chat(
                    question=question,
                    user_id=str(interaction.user.id),
                )

                logger.info(
                    "LLM completed | user=%s | answer_length=%s | answer_preview=%r",
                    interaction.user,
                    len(answer) if answer else 0,
                    answer[:100] if answer else None,
                )

        if not answer:
            logger.warning(
                "LLM returned empty answer | user=%s",
                interaction.user,
            )

            await interaction.followup.send(
                "I couldn't generate a response."
            )
            return

        logger.info(
            "Sending response to Discord | user=%s",
            interaction.user,
        )

        formatted_response = (
            f"**Question:** {question}\n"
            f"**Answer:** {answer}"
        )

        await send_long_response(
            interaction,
            formatted_response
)

        logger.info(
            "Discord response sent successfully | user=%s",
            interaction.user,
        )

    except TimeoutError:
        logger.warning(
            "LLM request timed out | user=%s | question=%r",
            interaction.user,
            question[:200],
        )

        await interaction.followup.send(
            "That request took too long. Try again."
        )

    except Exception:
        logger.exception(
            "/paimon_ask failed | user=%s | question=%r",
            interaction.user,
            question[:200],
        )

        try:
            await interaction.followup.send(
                "Something went wrong while generating the response."
            )
        except Exception:
            logger.exception(
                "Failed to send /paimon_ask error response to Discord"
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

    winner = random.choice([
        option_a,
        option_b
    ])

    loser = (
        option_b
        if winner == option_a
        else option_a
    )

    message_templates = [
        "**{winner} got the Victory Royale.**",
        "**{winner} got sent to Ram Ranch.**",
        "**{winner} is going to Brazil.**",
        "**{winner} did not win the 50/50.**",
        "**{winner} received a free trip to a Private Island™.**",
        "**{winner} received a mandatory invitation to a Diddy Party.**",
        "**{winner} won the 50/50 in the greatest gacha ever.**",
        "**{winner} was turned into a marketable plushie.**",
        "**{winner} became a Zenless Zone Zero stunner.**",
        "**{winner} was forced to update Prydwen for Honkai Star Rail.**",
        "**{winner} was forced to play Wuthering Waves.**",
        "**{winner} won a billion primogems (for real this time).**",
        "**{winner} gambled the house away.**",
        "**{winner} in fact, could not handle allat.**",
        "**{winner} vanished searching for Nobu.**",
        "**{winner} vanished searching for Sed.**",
        "**{winner} vanished searching for Loli.**",
        "**{winner} vanished searching for Shakkun.**",
        "**{winner} vanished searching for Zaizen.**",
        "**{winner} vanished searching for Omega.**",
        "**{winner} vanished searching for Ajoule.**",
        "**{winner} got lost looking for respawning chests.**",
        "**{winner}'s favorite live-service game announced EOS.**",
        "**{winner} opened a Mr. Beast Discord Image.**",
        "**{winner} got W Groomed at a Smash Ultimate tournament.**",
        "**{winner} got crushed playing Rock-Paper-Scissors with Dialyn.**",
        "**{winner} was disassembled by Grace Howard!**",
        "**{winner} got tag-teamed by Hu Tao, Sparkle, Burnice, and Yuzuha.**",
        "**{loser} ate Rina's cooking.**",
        "**{winner}'s favorite gacha did not make the top 10 of this month's revenue chart.**",
        "**{winner} was banished from Elder Yue's heavenly sect.**",
        "**{winner} won a lifetime supply of Jub's kebabs.**",
        "**{winner} courted death.**",
        "**{winner} did not receive the 5th aakek.**",
        "**{winner}'s Discord account was flagged for inappropriate content.**",
        "**{winner} is the Lebron James of randomly selected options.**",
        "**{winner} is the Bronny James of randomly selected options.**",
        "**{winner} was attacked by a kemonomimi in the woods. All of a sudden they have to take care of dozens of wolf children! Every night their number grows! Fail to do so and their partner, the Wolf Mama, will kill them! How many can they sustain?**", 
        "**Yumemizuki Mizuki will give {winner} good dreams tonight.**",
        "**Internet artists genderbent {winner}.**",
        "**Sending Scihub to {winner}'s location.**",
        "**If you goon, {winner} dies. Of course, we gooned.**",
        "**Get ready to learn Chinese, {winner}.**",
        "**Mihoyo satellite targeting system online: {winner} will be terminated in 15 seconds.**",
        "**Carl revealed {winner} is gay.**",
        "**Carl revealed {winner} is not gay.**",
        "**Gorlock sat on {loser}.**",
        "**Zhu Yuan sat on {loser}.**",
        "**Steam Status: {winner} is now playing: Femboy Futa House.**",
        "**Steam Status: {winner} is now playing: Sex with Hitler 3.**",
        "**Steam Status: {winner} is now playing: Showering with your Dad Simulator.**",
        "**Steam Status: {winner} is now playing: Super Lesbian Animal RPG.**",
        "**{winner} ejected {loser} from an airlock.**",
        "**{winner} fired {loser} into the sun.**",
        "**{winner} sent {loser} to #blue-archive.**",
        "**{winner} powercrept {loser} into T5.**",
        "**{winner} plapped {loser}.**",
        "**{winner} NTR'd all of {loser}'s waifus in Mudae.**",
   ]

    # Exclude any template used in the previous 10 duels
    eligible_ids = [
        index
        for index in range(len(message_templates))
        if index not in RECENT_DUEL_MESSAGE_IDS
    ]

    # Safety fallback if the pool ever becomes too small
    if not eligible_ids:
        eligible_ids = list(
            range(len(message_templates))
        )

    message_id = random.choice(
        eligible_ids
    )

    RECENT_DUEL_MESSAGE_IDS.append(
        message_id
    )

    result = message_templates[
        message_id
    ].format(
        winner=winner,
        loser=loser,
    )

    await ctx.reply(
        result
    )

    logger.info(
        "%s used !duel: %s vs %s -> %s | message_id=%s",
        ctx.author,
        option_a,
        option_b,
        winner,
        message_id,
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
    if message.author.bot or is_blocked_user(message.author.id):
        return

    triggered = False
    content = message.content.strip()

    # Dadbot feature
    dadmatch = re.search(
        r"\b(?:i['’]?m|i am)\s+(.+)",
        message.content,
        re.IGNORECASE,
    )

    if dadmatch and random.random() <= DADBOT_CHANCE:
        x = dadmatch.group(1).strip()

        if x:
            await message.reply(
                f"Hi {x}, I'm Paimon!"
            )
            triggered = True

    # Random Chinese translation through the LangGraph translation route
    if (
        not triggered
        and content
        and not content.startswith("!")
        and random.random() <= TRANSLATE_CHANCE
    ):
        try:
            translation = await run_translation(content)

            if translation:
                await message.reply(translation)
                triggered = True

        except Exception:
            logger.exception(
                "Random Chinese translation failed."
            )

    # Random reaction using custom emojis available in this server.
    if (
        not triggered
        and content
        and not content.startswith("!")
        and random.random() <= REACTION_CHANCE
    ):
        try:
            custom_emojis = [
                emoji
                for emoji in (message.guild.emojis if message.guild else [])
                if emoji.available
                and f"custom:{emoji.id}" not in RECENT_EMOJI_IDS
            ]

            fallback_emojis = [
                emoji
                for emoji in ["😂", "🌹", "💀", "🥀", "📮", "🫃"]
                if f"unicode:{emoji}" not in RECENT_EMOJI_IDS
            ]

            available_emojis = custom_emojis + fallback_emojis

            if not available_emojis:
                RECENT_EMOJI_IDS.clear()
                available_emojis = (
                    [
                        emoji
                        for emoji in (
                            message.guild.emojis
                            if message.guild
                            else []
                        )
                        if emoji.available
                    ]
                    + ["😂", "🌹", "💀", "🥀", "📮", "🫃"]
                )

            emoji = random.choice(available_emojis)

            if isinstance(emoji, discord.Emoji):
                RECENT_EMOJI_IDS.append(
                    f"custom:{emoji.id}"
                )
            else:
                RECENT_EMOJI_IDS.append(
                    f"unicode:{emoji}"
                )

            await message.add_reaction(emoji)
            triggered = True

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

    if isinstance(error, commands.CheckFailure):
        return

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