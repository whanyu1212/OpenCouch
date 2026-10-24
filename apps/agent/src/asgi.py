"""The app object Vercel serves (HY-352); Vercel needs an app, not a factory."""

from opencouch_agent.main import create_app

app = create_app()
