import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from first_agent import first_agent, infer_products_from_vague_input, InferredSetupSpecs
from spec_resolution_agent import spec_resolution_agent, resolve_specs

# ── Load API key ───────────────────────────────────────────────────────────
# main_agent.py lives at  backend/app/agents/main_agent.py
# .env lives at          backend/.env  (3 levels up from this file)
_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_env_path)

APP_NAME = "amazesearch_app"
USER_ID = "demo_user"
SESSION_ID = "demo_session"
SPEC_SESSION_ID = "demo_session_spec_resolution"


async def main():
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SPEC_SESSION_ID
    )

    # Runner for first_agent (the refinement loop)
    first_agent_runner = Runner(
        app_name=APP_NAME,
        agent=first_agent,
        session_service=session_service,
    )

    # Runner for spec_resolution_agent (search + structure pipeline)
    spec_runner = Runner(
        app_name=APP_NAME,
        agent=spec_resolution_agent,
        session_service=session_service,
    )

    print("Describe what you're setting up (e.g. 'beginner gym equipment').")
    print("Keep refining as needed. Type 'done' once you're satisfied with the list.\n")

    last_specs: InferredSetupSpecs | None = None

    while True:
        user_prompt = input("You: ").strip()

        if not user_prompt:
            continue

        if user_prompt.lower() in {"done", "exit", "quit", "q"}:
            break

        try:
            last_specs = await infer_products_from_vague_input(
                first_agent_runner, USER_ID, SESSION_ID, user_prompt
            )
        except Exception as e:
            print(f"\n[Error generating specs: {e}]\n")
            continue

        print("\nGuessed Setup Structure:")
        print(last_specs.model_dump_json(indent=2))
        print()

    if last_specs is None:
        print("\nNo confirmed product list to resolve. Exiting.")
        return

    print("\nResolving real-world specs and brands for your confirmed list...\n")

    try:
        resolved = await resolve_specs(
            spec_runner, USER_ID, SPEC_SESSION_ID, last_specs
        )
    except Exception as e:
        print(f"\n[Error resolving specs: {e}]")
        return

    print("Final Resolved Specs:")
    print(resolved.model_dump_json(indent=2))


if __name__ == "__main__":
    # Ensure your environment variable is set in backend/.env: GOOGLE_API_KEY="your-key"
    asyncio.run(main())