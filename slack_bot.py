"""Slack bot: answers policy questions when mentioned in a channel.

Uses the LLM with vector index (method 3). Runs locally with Socket Mode, so no public URL is needed.
Run:  py slack_bot.py   (stop with Ctrl+C)
"""
import re

from google import genai
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

import policy_engine as pe

client = genai.Client(api_key=pe.get_secret("GEMINI_API_KEY"))
policies = pe.load_policies()
index = pe.load_index(policies)
app = App(token=pe.get_secret("SLACK_BOT_TOKEN"))


@app.event("app_mention")
def on_mention(event, say, logger):
    question = re.sub(r"<@[A-Z0-9]+>", "", event.get("text", "")).strip()
    thread = event.get("thread_ts")  # reply in the thread only if the question was asked in one
    if not question:
        say(text="Ask me a policy question, for example: how many vacation days do I get?", thread_ts=thread)
        return
    try:
        result = pe.llm_with_index(client, question, policies, index)
        say(text=pe.format_for_slack(result), thread_ts=thread)
    except Exception as e:
        logger.exception(e)
        say(text=f"Sorry, I could not answer that right now ({str(e)[:150]}).", thread_ts=thread)


if __name__ == "__main__":
    print("Policy bot running. Mention it in Slack. Ctrl+C to stop.")
    SocketModeHandler(app, pe.get_secret("SLACK_APP_TOKEN")).start()
