#!/bin/bash
cd "/Users/themastermind/mflux"
python3 -m venv .venv && source .venv/bin/activate
pip install -U -r scripts/requirements.txt 
python3 scripts/discord_bot.py