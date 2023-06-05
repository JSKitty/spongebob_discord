import discord
from discord.ext import commands
import re
import asyncio
import random
import os
import logging

from fgk_spongebot import FGKBot

# Set up logging for error handling
logging.basicConfig(level=logging.ERROR, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

future_gadget_kr3w = FGKBot(bot)

@bot.event
async def on_ready():
    try:
        print('We have logged in as {0.user}'.format(bot))
    except Exception as e:
        logging.error(f"Error in on_ready: {e}")

async def run_avatar_waifu(ctx, query):
    try:
        await future_gadget_kr3w.avatar_waifu(ctx, query, ctx.message.author.name, ctx.message.author.id, ctx.message.channel.id)
    except Exception as e:
        logging.error(f"Error in sb command: {e}")

@bot.command(name='sb')
async def spongebob(ctx, *, query=""):
    try:
        if ctx.message.author == bot.user:
            return

        async with ctx.typing():
            asyncio.create_task(run_avatar_waifu(ctx, query))
    except Exception as e:
        logging.error(f"Error in sb command: {e}")

try:
    loop = asyncio.get_event_loop()
    loop.create_task(bot.start(future_gadget_kr3w.BOT_CONFIG['DISCORD-TOKEN']))
    loop.run_forever()
except Exception as e:
    logging.error(f"Error running bot: {e}")
